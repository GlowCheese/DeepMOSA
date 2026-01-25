####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("a: 1\nb: 2\nc: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2, "c": 3}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=12)

def test_tokenize_yaml_nested_dict():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=14)

def test_tokenize_yaml_nested_list():
    token = tokenize_yaml("- - 1\n  - 2\n- - 3\n  - 4")
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=5, char_index=20)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("a: b: c")
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"
        assert exc.position == Position(line_no=1, column_no=6, char_index=5)


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    content = "hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    content = "123"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    content = "123.45"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool():
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    content = "[1, 2, 3]"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    content = "{a: 1, b: 2}"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_dict():
    content = "{a: {b: 2}}"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 2}}
    assert token.string == "{a: {b: 2}}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_list():
    content = "[1, [2, 3]]"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, [2, 3]]
    assert token.string == "[1, [2, 3]]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_invalid_yaml():
    content = "{a: 1, b: 2"
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.text == "did not find expected ',' or '}'."
        assert e.code == "parse_error"
        assert e.position.line_no == 1
        assert e.position.column_no == 11
        assert e.position.char_index == 10


# LLM-generated content at query #3
#--------------------------

```python
def test_yaml_is_not_installed():
    yaml = None
    result = tokenize_yaml("")
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml_empty_content():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #5
#--------------------------

def test_validate_yaml_with_valid_yaml():
    content = "name: John\nage: 30"
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_invalid_yaml_syntax():
    content = "name: John\nage:"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "could not find expected ':'" in exc.text.lower()

def test_validate_yaml_with_validation_error():
    content = "name: John\nage: thirty"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as exc:
        assert any(msg.code == "type" for msg in exc.messages())

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_with_bytes_input():
    content = b"name: John\nage: 30"
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_nested_structure():
    content = "person:\n  name: John\n  age: 30"
    validator = Schema(fields={"person": Schema(fields={"name": Field(), "age": Field()})})
    result = validate_yaml(content, validator)
    assert result == {"person": {"name": "John", "age": 30}}

def test_validate_yaml_with_missing_required_field():
    content = "name: John"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as exc:
        assert any(msg.code == "required" for msg in exc.messages())

def test_validate_yaml_with_default_values():
    content = "name: John"
    validator = Schema(fields={"name": Field(), "age": Field(default=30)})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "key: value"
    validator = Schema(fields={"key": Field()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_input():
    content = "key: 123"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert error.messages()[0].code == "type"

def test_validate_yaml_empty_string():
    content = ""
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "no_content"

def test_validate_yaml_invalid_yaml_syntax():
    content = "key: value:"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "parse_error"

def test_validate_yaml_null_value():
    content = "key: null"
    validator = Schema(fields={"key": Field(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}


# LLM-generated content at query #7
#--------------------------

def test_tokenize_yaml_handles_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #8
#--------------------------

def test_tokenize_yaml_handles_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_yaml_with_pyyaml_installed():
    validate_yaml("key: value", Field())


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml_with_valid_input():
    content = "name: John\nage: 30"
    field = Field()
    result = validate_yaml(content, field)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_empty_content():
    content = ""
    field = Field()
    try:
        validate_yaml(content, field)
    except ParseError as error:
        assert error.text == "No content."
        assert error.code == "no_content"

def test_validate_yaml_with_invalid_yaml():
    content = "name: John\n: 30"
    field = Field()
    try:
        validate_yaml(content, field)
    except ParseError as error:
        assert error.code == "parse_error"

def test_validate_yaml_with_schema_validation_error():
    content = "name: John\nage: thirty"
    schema = Schema(fields={"age": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as error:
        assert len(error.messages) == 1
        assert error.messages[0].code == "type"

def test_validate_yaml_with_required_field_error():
    content = "name: John"
    schema = Schema(fields={"age": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as error:
        assert len(error.messages) == 1
        assert error.messages[0].code == "required"
        assert error.messages[0].index == ["age"]

def test_validate_yaml_with_bytes_input():
    content = b"name: John\nage: 30"
    field = Field()
    result = validate_yaml(content, field)
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_yaml_handles_parse_error_without_problem():
    import pytest

    try:
        tokenize_yaml("invalid yaml")
    except Exception as exc:
        assert hasattr(exc, 'problem') and exc.problem is None


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_yaml_empty_content():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError:
        pass
    else:
        assert False, "Expected ParseError to be raised for empty content"


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_assert_problem_not_none():
    yaml_content = "invalid: yaml: content"
    try:
        tokenize_yaml(yaml_content)
    except Exception as exc:
        assert exc.problem is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    content = ""
    try:
        tokenize_yaml(content)
        assert False, "Expected ParseError to be raised"
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_list():
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.string == "- item1\n- item2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=14)
    assert token.value == ["item1", "item2"]

def test_tokenize_yaml_dict():
    content = "key1: value1\nkey2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key1: value1\nkey2: value2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=10, char_index=23)
    assert token.value == {"key1": "value1", "key2": "value2"}

def test_tokenize_yaml_int():
    content = "key: 123"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key: 123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)
    assert token.value == {"key": 123}

def test_tokenize_yaml_float():
    content = "key: 123.45"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key: 123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.value == {"key": 123.45}

def test_tokenize_yaml_bool():
    content = "key: true"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key: true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.value == {"key": True}

def test_tokenize_yaml_null():
    content = "key: null"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.string == "key: null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.value == {"key": None}


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_yaml_pyyaml_not_installed():
    try:
        import sys
        sys.modules['yaml'] = None
        tokenize_yaml("")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        del sys.modules['yaml']


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml_empty_content():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)


def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)


def test_tokenize_yaml_dict():
    token = tokenize_yaml("a: 1\nb: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "a: 1\nb: 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=7)


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  - 1\n  - 2\nb: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": 3}
    assert token.string == "a:\n  - 1\n  - 2\nb: 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=3, char_index=15)


def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_invalid_syntax():
    try:
        tokenize_yaml(": invalid")
    except ParseError as exc:
        assert "could not find expected key" in exc.text
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1


# LLM-generated content at query #18
#--------------------------

def test_validate_yaml_with_valid_yaml():
    content = "key: value"
    field = Field()
    result = validate_yaml(content, field)
    assert result == {"key": "value"}

def test_validate_yaml_with_invalid_yaml():
    content = "key: value\ninvalid"
    field = Field()
    try:
        validate_yaml(content, field)
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"

def test_validate_yaml_with_empty_content():
    content = ""
    field = Field()
    try:
        validate_yaml(content, field)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

def test_validate_yaml_with_validation_error():
    content = "key: 123"
    schema = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"

def test_validate_yaml_with_required_field_error():
    content = "{}"
    schema = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'key' is required."
        assert messages[0].code == "required"

def test_validate_yaml_with_positional_validation_error():
    content = "key: 123"
    schema = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == len(content) - 1


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except Exception as exc:
        assert str(exc) == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)


def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == "- 1\n- 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=5)


def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=13)


def test_tokenize_yaml_invalid_syntax():
    try:
        tokenize_yaml("key: [")
    except Exception as exc:
        assert "did not find expected node content" in str(exc)
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    import yaml
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml

    invalid_yaml = "key: value\nkey2: value2\n\tkey3: value3"
    try:
        tokenize_yaml(invalid_yaml)
    except Exception as exc:
        assert exc.problem_mark is not None


# LLM-generated content at query #21
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == "- 1\n- 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=5)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=13)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [1, 2")
    except ParseError as exc:
        assert "did not find expected ']'" in exc.text
        assert exc.code == "parse_error"
        assert isinstance(exc.position, Position)


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml_with_valid_data():
    content = "name: John Doe\nage: 30"
    field = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, field)
    assert result == {"name": "John Doe", "age": 30}

def test_validate_yaml_with_missing_required_field():
    content = "name: John Doe"
    field = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, field)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]

def test_validate_yaml_with_invalid_data_type():
    content = "name: John Doe\nage: thirty"
    field = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, field)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Did not match any valid type."
        assert messages[0].code == "union"
        assert messages[0].index == ["age"]

def test_validate_yaml_with_empty_content():
    content = ""
    field = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, field)
    except ParseError as error:
        assert error.text == "No content."
        assert error.code == "no_content"
        assert error.position.line_no == 1
        assert error.position.column_no == 1
        assert error.position.char_index == 0

def test_validate_yaml_with_invalid_yaml_syntax():
    content = "name: John Doe\nage:"
    field = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, field)
    except ParseError as error:
        assert error.text == "could not find expected ':'."
        assert error.code == "parse_error"


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_yaml_empty_content_raises_parse_error():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.string == "hello"
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.string == "42"
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)


def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.string == "3.14"
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.string == "true"
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.string == "null"
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.string == "- 1\n- 2\n- 3"
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)


def test_tokenize_yaml_dict():
    token = tokenize_yaml("a: 1\nb: 2\nc: 3")
    assert isinstance(token, DictToken)
    assert token.string == "a: 1\nb: 2\nc: 3"
    assert token.value == {"a": 1, "b": 2, "c": 3}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=12)


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  - b: 1\n    c: 2\nd: 3")
    assert isinstance(token, DictToken)
    assert token.string == "a:\n  - b: 1\n    c: 2\nd: 3"
    assert token.value == {"a": [{"b": 1, "c": 2}], "d": 3}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=3, char_index=22)


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml_installed():
    import sys
    sys.modules["yaml"] = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #26
#--------------------------

def test_tokenize_yaml_handles_parse_error_without_problem_mark():
    try:
        tokenize_yaml("invalid: [")
    except Exception as exc:
        assert not isinstance(exc, AssertionError), "Should not raise AssertionError for None problem_mark"


# LLM-generated content at query #27
#--------------------------

def test_validate_yaml_with_valid_yaml():
    content = "key: value"
    validator = Schema(fields={"key": Field()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_invalid_yaml_parse_error():
    content = "key: value\ninvalid"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as error:
        assert error.code == "parse_error"
        assert error.position.line_no == 2

def test_validate_yaml_with_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as error:
        assert len(error.messages()) > 0
        assert error.messages()[0].code == "type"

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as error:
        assert error.code == "no_content"
        assert error.position.line_no == 1

def test_validate_yaml_with_bytes_input():
    content = b"key: value"
    validator = Schema(fields={"key": Field()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_required_field_error():
    content = "{}"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as error:
        assert len(error.messages()) > 0
        assert error.messages()[0].code == "required"


# LLM-generated content at query #28
#--------------------------

def test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("test")
    except AssertionError as exc:
        assert str(exc) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        assert exc.problem is None


# LLM-generated content at query #30
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #31
#--------------------------

def test_validate_yaml_does_not_raise_when_pyyaml_is_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = object()
    try:
        validate_yaml(content="", validator=Field())
    except AssertionError:
        assert False, "validate_yaml should not raise when yaml is installed"
    finally:
        if original_yaml is not None:
            sys.modules['yaml'] = original_yaml
        else:
            del sys.modules['yaml']


# LLM-generated content at query #32
#--------------------------

def test_tokenize_yaml_handles_parse_error_without_problem_mark():
    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #33
#--------------------------

```
def test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_token():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int_token():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float_token():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool_token():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null_token():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list_token():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)

def test_tokenize_yaml_dict_token():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as e:
        assert "could not find expected ':'" in e.text
        assert e.code == "parse_error"
        assert e.position == Position(line_no=2, column_no=1, char_index=11)


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    content = "name: John\nage: 30"
    schema = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_missing_required_field():
    content = "name: John"
    schema = Schema(fields={"name": Field(), "age": Field()})
    error = None
    try:
        validate_yaml(content, schema)
    except ValidationError as e:
        error = e
    assert error is not None
    assert len(error.messages) == 1
    assert error.messages[0].text == "The field 'age' is required."

def test_validate_yaml_with_invalid_yaml():
    content = "name: John\nage:"
    schema = Schema(fields={"name": Field(), "age": Field()})
    error = None
    try:
        validate_yaml(content, schema)
    except ParseError as e:
        error = e
    assert error is not None
    assert error.text == "No value after 'age:'.", "Expected error message for invalid YAML"

def test_validate_yaml_with_null_value_and_allow_null():
    content = "name: null"
    schema = Schema(fields={"name": Field(allow_null=True)})
    result = validate_yaml(content, schema)
    assert result == {"name": None}

def test_validate_yaml_with_null_value_and_disallow_null():
    content = "name: null"
    schema = Schema(fields={"name": Field(allow_null=False)})
    error = None
    try:
        validate_yaml(content, schema)
    except ValidationError as e:
        error = e
    assert error is not None
    assert len(error.messages) == 1
    assert error.messages[0].text == "May not be null."


# LLM-generated content at query #3
#--------------------------

def test_tokenize_yaml_empty_content_raises_parse_error():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml_installed():
    yaml = None
    try:
        tokenize_yaml("key: value")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=10)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=14)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index == 11


# LLM-generated content at query #6
#--------------------------

Here are the test cases for the `validate_yaml` function:


# LLM-generated content at query #7
#--------------------------

def test_tokenize_yaml_handles_parse_error_without_problem_mark():
    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        assert exc.problem_mark is None


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_yaml_pyyaml_not_installed():
    import sys
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("content")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_yaml_without_problem_mark():
    yaml_content = b'key: value'
    try:
        tokenize_yaml(yaml_content)
    except Exception as exc:
        assert exc.problem_mark is None


# LLM-generated content at query #11
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)
    assert token.string == "hello"


def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)
    assert token.string == "123"


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)
    assert token.string == "123.45"


def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)
    assert token.string == "true"


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)
    assert token.string == "null"


def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)
    assert token.string == "- 1\n- 2\n- 3"


def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)
    assert token.string == "key: value"


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=13)
    assert token.string == "key:\n  - 1\n  - 2"


def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)
    assert token.string == "hello"


def test_tokenize_yaml_invalid_syntax():
    try:
        tokenize_yaml("key: : value")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "could not find expected ':'" in exc.text.lower()


# LLM-generated content at query #12
#--------------------------

def test_validate_yaml_without_pyyaml_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        validate_yaml("", Field())
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #13
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)


def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)


def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=14)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml_with_valid_yaml():
    content = "name: John\nage: 30"
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_invalid_yaml():
    content = "name: John\nage: thirty"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].code == "type"

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "no_content"
        assert error.position.line_no == 1
        assert error.position.column_no == 1
        assert error.position.char_index == 0

def test_validate_yaml_with_missing_required_field():
    content = "name: John"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].code == "required"
        assert error.messages()[0].index == ["age"]

def test_validate_yaml_with_invalid_field_type():
    content = "name: John\nage: thirty"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].code == "type"
        assert error.messages()[0].index == ["age"]

def test_validate_yaml_with_nested_schema():
    content = "person:\n  name: John\n  age: 30"
    validator = Schema(fields={"person": Schema(fields={"name": Field(), "age": Field()})})
    result = validate_yaml(content, validator)
    assert result == {"person": {"name": "John", "age": 30}}


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_yaml_handles_parse_error_without_problem_mark():
    try:
        tokenize_yaml("invalid: yaml: content")
    except Exception as exc:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem_mark():
    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        assert exc.problem_mark is None


# LLM-generated content at query #17
#--------------------------

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == "- 1\n- 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=5)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [1, 2")
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index > 0


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_with_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_with_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)


def test_tokenize_yaml_with_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)


def test_tokenize_yaml_with_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_with_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_with_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)


def test_tokenize_yaml_with_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)


def test_tokenize_yaml_with_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=14)


def test_tokenize_yaml_with_invalid_yaml():
    try:
        tokenize_yaml("key: [1, 2")
    except ParseError as exc:
        assert "did not find expected ',' or ']'" in exc.text
        assert exc.code == "parse_error"


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml_assertion_false():
    try:
        validate_yaml("", Field())
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    try:
        tokenize_yaml("invalid: yaml: content")
    except Exception as exc:
        assert exc.problem is None


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    content = "invalid_yaml_content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        exc.problem = None
        exc.problem_mark = None
        try:
            tokenize_yaml(content)
        except Exception as e:
            pass


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "name: John\nage: 25"
    schema = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 25}

def test_validate_yaml_invalid_input():
    content = "name: John\nage: twenty-five"
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"

def test_validate_yaml_missing_required_field():
    content = "name: John"
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"

def test_validate_yaml_empty_content():
    content = ""
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, schema)
    except ParseError as error:
        assert error.code == "no_content"
        assert error.position.line_no == 1
        assert error.position.column_no == 1
        assert error.position.char_index == 0

def test_validate_yaml_invalid_yaml():
    content = "name: John\nage: 25\ninvalid"
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, schema)
    except ParseError as error:
        assert error.code == "parse_error"


# LLM-generated content at query #24
#--------------------------

```
def test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == "- 1\n- 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=6)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [1, 2")
    except ParseError as exc:
        assert "could not find expected ']'" in exc.text
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1
        assert exc.position.column_no >= 1


# LLM-generated content at query #26
#--------------------------

def test_validate_yaml_with_pyyaml_installed():
    import sys
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = object()
    try:
        from typesystem.tokenize.tokenize_yaml import validate_yaml
        validate_yaml(content="", validator=Field())
    finally:
        if original_yaml is None:
            del sys.modules['yaml']
        else:
            sys.modules['yaml'] = original_yaml


# LLM-generated content at query #27
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == "123.45"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_scalar_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2\n- 3")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "- 1\n- 2\n- 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("a: 1\nb: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "a: 1\nb: 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=7)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  - 1\n  - 2\nb: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": 3}
    assert token.string == "a:\n  - 1\n  - 2\nb: 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=3, char_index=15)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_syntax():
    try:
        tokenize_yaml("a: b: c")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index > 0


# LLM-generated content at query #28
#--------------------------

```python
def test_tokenize_yaml_handles_parse_error_without_problem_mark():
    try:
        tokenize_yaml("invalid: yaml: content")
    except ParseError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == 1
    assert token.value[1] == 2
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=5)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [1, 2")
    except ParseError as exc:
        assert exc.text.startswith("did not find expected node content")
        assert exc.code == "parse_error"


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    validator = Schema(fields={"name": Field()})
    content = "name: John"
    result = validate_yaml(content, validator)
    assert result == {"name": "John"}

def test_validate_yaml_with_invalid_content():
    validator = Schema(fields={"name": Field()})
    content = "name: 123"
    result = validate_yaml(content, validator)
    assert isinstance(result, ValidationError)

def test_validate_yaml_with_empty_content():
    validator = Schema(fields={"name": Field()})
    content = ""
    result = validate_yaml(content, validator)
    assert isinstance(result, ParseError)

def test_validate_yaml_with_invalid_yaml():
    validator = Schema(fields={"name": Field()})
    content = "name: {"
    result = validate_yaml(content, validator)
    assert isinstance(result, ParseError)

def test_validate_yaml_with_missing_required_field():
    validator = Schema(fields={"name": Field()})
    content = "age: 30"
    result = validate_yaml(content, validator)
    assert isinstance(result, ValidationError)

def test_validate_yaml_with_null_value():
    validator = Schema(fields={"name": Field(allow_null=True)})
    content = "name: null"
    result = validate_yaml(content, validator)
    assert result == {"name": None}


# LLM-generated content at query #31
#--------------------------

```python
def test_tokenize_yaml_scalar():
    content = "value"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float():
    content = "3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_bool():
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_null():
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=6, char_index=13)

def test_tokenize_yaml_dict():
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested():
    content = "key:\n  nested_key: nested_value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested_key": "nested_value"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=20, char_index=28)

def test_tokenize_yaml_empty_string():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_invalid_yaml():
    content = "key: value\ninvalid"
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- 1\n- 2")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=4, char_index=6)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_dict():
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested": "value"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=15, char_index=18)

def test_tokenize_yaml_nested_list():
    token = tokenize_yaml("- - 1\n- - 2")
    assert isinstance(token, ListToken)
    assert token.value == [[1], [2]]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=6, char_index=9)

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: : value")
    except ParseError as exc:
        assert "could not find expected ':' while scanning a simple key" in exc.text
        assert exc.code == "parse_error"


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        assert exc.problem is None


