####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - two")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, "two"]}
    assert token.string == "key:\n  - 1\n  - two"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=6, char_index=17)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0


# LLM-generated content at query #2
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  \t  ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_simple_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_integer():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value\nanother: 123")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "another": 123}
    assert token.string == "key: value\nanother: 123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=10, char_index=23)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("list:\n  - 1\n  - 2\ndict:\n  key: val")
    assert isinstance(token, DictToken)
    assert token.value == {"list": [1, 2], "dict": {"key": "val"}}
    assert token.string == "list:\n  - 1\n  - 2\ndict:\n  key: val"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=5, column_no=9, char_index=34)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"test: data")
    assert isinstance(token, DictToken)
    assert token.value == {"test": "data"}
    assert token.string == "test: data"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.line_no >= 1

def test_tokenize_yaml_lookup_in_dict():
    token = tokenize_yaml("a: 1\nb: 2")
    child = token.lookup(["b"])
    assert isinstance(child, ScalarToken)
    assert child.value == 2
    assert child.string == "2"
    assert child.start == Position(line_no=2, column_no=4, char_index=7)
    assert child.end == Position(line_no=2, column_no=4, char_index=7)

def test_tokenize_yaml_lookup_in_list():
    token = tokenize_yaml("- x\n- y\n- z")
    child = token.lookup([1])
    assert isinstance(child, ScalarToken)
    assert child.value == "y"
    assert child.string == "y"
    assert child.start == Position(line_no=2, column_no=3, char_index=5)
    assert child.end == Position(line_no=2, column_no=3, char_index=5)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #3
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    yaml_content = "key: !!invalid_tag value"
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(yaml_content)
    except Exception as exc:
        pass


# LLM-generated content at query #4
#--------------------------

def test_tokenize_yaml_assertion_fails_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    sys.modules['yaml'] = None
    try:
        module.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        del sys.modules['yaml']


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem_mark():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    import pytest
    from typesystem.base import Position
    from typesystem.exceptions import ParseError

    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self, problem, problem_mark):
            self.problem = problem
            self.problem_mark = problem_mark

    content = "invalid: ["
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(content)
    except Exception as e:
        assert isinstance(e, AssertionError)


# LLM-generated content at query #6
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    import pytest
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = None
            self.problem_mark = None
    with pytest.raises(AssertionError):
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("invalid: [")


# LLM-generated content at query #7
#--------------------------

def test_validate_yaml_valid_yaml():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    class TestField(Field):
        def validate(self, value):
            return value
    field = TestField()
    content = "key: value"
    result = validate_yaml(content, field)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml_parse_error():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    from typesystem.exceptions import ParseError
    class TestField(Field):
        def validate(self, value):
            return value
    field = TestField()
    content = "key: ["
    try:
        validate_yaml(content, field)
    except ParseError as exc:
        assert exc.code == "parse_error"

def test_validate_yaml_empty_content():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    from typesystem.exceptions import ParseError
    class TestField(Field):
        def validate(self, value):
            return value
    field = TestField()
    content = ""
    try:
        validate_yaml(content, field)
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_validation_error_with_positions():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.exceptions import ValidationError
    class RequiredField(Field):
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value
    schema = Schema(fields={"key": RequiredField()})
    content = "other: value"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["key"]
        assert msg.start_position.char_index == 0

def test_validate_yaml_bytes_input():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    class TestField(Field):
        def validate(self, value):
            return value
    field = TestField()
    content = b"key: value"
    result = validate_yaml(content, field)
    assert result == {"key": "value"}

def test_validate_yaml_with_schema_validation():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    schema = Schema(fields={"age": Integer()})
    content = "age: 25"
    result = validate_yaml(content, schema)
    assert result == {"age": 25}

def test_validate_yaml_with_schema_validation_error():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.exceptions import ValidationError
    schema = Schema(fields={"age": Integer()})
    content = "age: invalid"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) > 0
        msg = messages[0]
        assert msg.code == "integer"
        assert msg.index == ["age"]

def test_validate_yaml_nested_structure_validation():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    inner_schema = Schema(fields={"count": Integer()})
    outer_schema = Schema(fields={"data": inner_schema, "name": String()})
    content = "data:\n  count: 5\nname: test"
    result = validate_yaml(content, outer_schema)
    assert result == {"data": {"count": 5}, "name": "test"}

def test_validate_yaml_nested_structure_validation_error():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.exceptions import ValidationError
    inner_schema = Schema(fields={"count": Integer()})
    outer_schema = Schema(fields={"data": inner_schema, "name": String()})
    content = "data:\n  count: invalid\nname: test"
    try:
        validate_yaml(content, outer_schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) > 0
        msg = messages[0]
        assert msg.code == "integer"
        assert msg.index == ["data", "count"]


# LLM-generated content at query #8
#--------------------------

def test_validate_yaml_raises_assertion_error_when_yaml_is_none():
    yaml = None
    content = "key: value"
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except AssertionError as error:
        assert str(error) == "'pyyaml' must be installed."


# LLM-generated content at query #9
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=15)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "did not find expected node content" in e.text

def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- a\n- b")
    child = token.lookup([0])
    assert isinstance(child, ScalarToken)
    assert child.value == "a"
    assert child.string == "a"
    assert child.start == Position(line_no=1, column_no=3, char_index=2)
    assert child.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  \t  ")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_simple_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"

def test_tokenize_yaml_integer():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"

def test_tokenize_yaml_boolean_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"

def test_tokenize_yaml_boolean_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"

def test_tokenize_yaml_simple_list():
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

def test_tokenize_yaml_nested_list():
    token = tokenize_yaml("- [a, b]\n- [c, d]")
    assert isinstance(token, ListToken)
    assert token.value == [["a", "b"], ["c", "d"]]

def test_tokenize_yaml_simple_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_nested_dict():
    token = tokenize_yaml("outer:\n  inner: nested")
    assert isinstance(token, DictToken)
    assert token.value == {"outer": {"inner": "nested"}}

def test_tokenize_yaml_mixed_structure():
    token = tokenize_yaml("list:\n  - a\n  - b\ndict:\n  key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["a", "b"], "dict": {"key": "value"}}

def test_tokenize_yaml_positions_simple():
    content = "key: value"
    token = tokenize_yaml(content)
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_positions_multiline():
    content = "key:\n  nested: value"
    token = tokenize_yaml(content)
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=14, char_index=19)

def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_yaml_lookup_nested():
    token = tokenize_yaml("outer:\n  inner: nested")
    child = token.lookup(["outer", "inner"])
    assert isinstance(child, ScalarToken)
    assert child.value == "nested"

def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- first\n- second")
    child = token.lookup([0])
    assert isinstance(child, ScalarToken)
    assert child.value == "first"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code

def test_tokenize_yaml_scanner_error():
    try:
        tokenize_yaml("key: @invalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.code

def test_tokenize_yaml_string_property():
    token = tokenize_yaml("hello world")
    assert token.string == "hello world"

def test_tokenize_yaml_equality():
    token1 = tokenize_yaml("key: value")
    token2 = tokenize_yaml("key: value")
    assert token1 == token2

def test_tokenize_yaml_repr():
    token = tokenize_yaml("test")
    assert repr(token) == "ScalarToken('test')"


# LLM-generated content at query #11
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.string == "a:\n  b: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=11, char_index=13)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_multiline_scalar():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=18)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\n*anchor")
    assert isinstance(token, ListToken)
    assert token.value == ["value", "value"]
    assert token.string == "&anchor value\n*anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=19)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_complex_structure():
    yaml_content = """
    - name: John
      age: 30
      hobbies:
        - reading
        - hiking
    - name: Jane
      age: 25
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0]["name"] == "John"
    assert token.value[0]["age"] == 30
    assert token.value[0]["hobbies"] == ["reading", "hiking"]
    assert token.value[1]["name"] == "Jane"
    assert token.value[1]["age"] == 25

def test_tokenize_yaml_token_equality():
    token1 = tokenize_yaml("hello")
    token2 = tokenize_yaml("hello")
    assert token1 == token2
    token3 = tokenize_yaml("world")
    assert token1 != token3

def test_tokenize_yaml_scalar_hash():
    token = tokenize_yaml("hello")
    hash_value = hash(token)
    assert isinstance(hash_value, int)


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml_empty_string():
    from typesystem.fields import Field
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError

    class TestField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    field = TestField()
    try:
        validate_yaml("", field)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_validate_yaml_valid_simple_value():
    from typesystem.fields import Field
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    field = TestField()
    value = validate_yaml("hello", field)
    assert value == "hello"

def test_validate_yaml_invalid_simple_value():
    from typesystem.fields import Field
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    class TestField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    field = TestField()
    try:
        validate_yaml("123", field)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].text == "Must be a string."
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 2

def test_validate_yaml_valid_object():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestSchema(Schema):
        fields = {"age": Integer()}

    schema = TestSchema()
    value = validate_yaml("age: 25", schema)
    assert value == {"age": 25}

def test_validate_yaml_invalid_object_missing_required():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    class TestSchema(Schema):
        fields = {"age": Integer()}

    schema = TestSchema()
    try:
        validate_yaml("name: John", schema)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].text == "This field is required."
        assert messages[0].index == ["age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 10

def test_validate_yaml_invalid_object_wrong_type():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    class TestSchema(Schema):
        fields = {"age": Integer()}

    schema = TestSchema()
    try:
        validate_yaml("age: twenty", schema)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == ["age"]
        assert messages[0].start_position.char_index == 5
        assert messages[0].end_position.char_index == 11

def test_validate_yaml_valid_list():
    from typesystem.fields import Array, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    field = Array(items=Integer())
    value = validate_yaml("- 1\n- 2\n- 3", field)
    assert value == [1, 2, 3]

def test_validate_yaml_invalid_list_item():
    from typesystem.fields import Array, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    field = Array(items=Integer())
    try:
        validate_yaml("- 1\n- two\n- 3", field)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == [1]
        assert messages[0].start_position.char_index == 4
        assert messages[0].end_position.char_index == 7

def test_validate_yaml_bytes_input():
    from typesystem.fields import Field
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    field = TestField()
    value = validate_yaml(b"hello", field)
    assert value == "hello"

def test_validate_yaml_yaml_parse_error():
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError

    from typesystem.fields import Field
    class TestField(Field):
        def validate(self, value):
            return value

    field = TestField()
    try:
        validate_yaml(": invalid", field)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "could not find expected key" in exc.text.lower()
        assert exc.position.char_index >= 0

def test_validate_yaml_nested_object_validation():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    class PersonSchema(Schema):
        fields = {"name": String(), "age": Integer()}

    class CompanySchema(Schema):
        fields = {"name": String(), "ceo": PersonSchema()}

    schema = CompanySchema()
    try:
        validate_yaml("name: Acme\nceo:\n  name: John\n  age: forty", schema)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == ["ceo", "age"]
        assert messages[0].start_position.char_index == 28
        assert messages[0].end_position.char_index == 33

def test_validate_yaml_with_default_values():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestSchema(Schema):
        fields = {
            "name": String(default="Unknown"),
            "age": Integer()
        }

    schema = TestSchema()
    value = validate_yaml("age: 30", schema)
    assert value == {"name": "Unknown", "age": 30}

def test_validate_yaml_with_read_only_field():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestSchema(Schema):
        fields = {
            "id": Integer(read_only=True),
            "name": String()
        }

    schema = TestSchema()
    value = validate_yaml("name: John\nid: 123", schema)
    assert value == {"name": "John"}

def test_validate_yaml_allow_null():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml

    class TestSchema(Schema):
        fields = {"name": String(), "age": Integer()}

    schema = TestSchema(allow_null=True)
    value = validate_yaml("null", schema)
    assert value is None

def test_validate_yaml_not_allow_null():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError

    class TestSchema(Schema):
        fields = {"name": String(), "age": Integer()}

    schema = TestSchema(allow_null=False)
    try:
        validate_yaml("null", schema)
        assert False, "


# LLM-generated content at query #13
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

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

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=15)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("|\n  hello\n  world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello\nworld\n"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=19)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml_successful_validation():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "42"
    result = validate_yaml(content, validator)
    assert result == 42

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "not_an_integer"
    try:
        validate_yaml(content, validator)
        assert False
    except Exception as error:
        assert hasattr(error, 'messages')
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == 'type'
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_yaml_empty_content():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = ""
    try:
        validate_yaml(content, validator)
        assert False
    except Exception as error:
        assert error.code == 'no_content'
        assert error.position is not None

def test_validate_yaml_parse_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "invalid: [unclosed"
    try:
        validate_yaml(content, validator)
        assert False
    except Exception as error:
        assert error.code == 'parse_error'
        assert error.position is not None

def test_validate_yaml_with_schema():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String(), "age": Integer()}
    validator = Schema(fields=fields)
    content = "name: John\nage: 30"
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_schema_validation_error():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String(), "age": Integer()}
    validator = Schema(fields=fields)
    content = "name: John\nage: not_an_integer"
    try:
        validate_yaml(content, validator)
        assert False
    except Exception as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == 'type'
        assert message.index == ['age']

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = String()
    content = b"hello"
    result = validate_yaml(content, validator)
    assert result == "hello"

def test_validate_yaml_complex_structure():
    from typesystem.fields import String, Integer, Array
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    item_schema = Schema(fields={"name": String(), "value": Integer()})
    validator = Array(items=item_schema)
    content = "- name: item1\n  value: 10\n- name: item2\n  value: 20"
    result = validate_yaml(content, validator)
    assert result == [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}]

def test_validate_yaml_null_value():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = String(allow_null=True)
    content = "null"
    result = validate_yaml(content, validator)
    assert result is None

def test_validate_yaml_with_union_field():
    from typesystem.fields import String, Integer, Union
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Union(any_of=[String(), Integer()])
    content = "123"
    result = validate_yaml(content, validator)
    assert result == 123
    content = '"hello"'
    result = validate_yaml(content, validator)
    assert result == "hello"


# LLM-generated content at query #15
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    yaml_content = "invalid: yaml: content"
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(yaml_content)
    except Exception as exc:
        pass


# LLM-generated content at query #16
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  \t  ")
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
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.string == "a:\n  b: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=11, char_index=14)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml(": invalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index >= 0

def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- a\n- b")
    child = token.lookup([0])
    assert isinstance(child, ScalarToken)
    assert child.value == "a"
    assert child.string == "a"
    assert child.start == Position(line_no=1, column_no=3, char_index=2)
    assert child.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError

    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = None
            self.problem_mark = None

    try:
        tokenize_yaml("invalid: [")
    except Exception as e:
        assert isinstance(e, AssertionError)


# LLM-generated content at query #18
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    yaml.scanner.ScannerError = type('ScannerError', (Exception,), {'problem': 'test', 'problem_mark': None})
    exc = yaml.scanner.ScannerError('test')
    exc.problem = 'test'
    exc.problem_mark = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml('invalid: @')
    except yaml.scanner.ScannerError:
        pass


# LLM-generated content at query #19
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        result = e
    assert result.code == "no_content"
    assert result.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested": "value"}}

def test_tokenize_yaml_mixed_structure():
    token = tokenize_yaml("list:\n  - item1\n  - item2")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"]}

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [")
    except ParseError as e:
        result = e
    assert result.code == "parse_error"

def test_tokenize_yaml_token_positions():
    token = tokenize_yaml("key: value")
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_token_string():
    token = tokenize_yaml("key: value")
    assert token.string == "key: value"

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"


# LLM-generated content at query #20
#--------------------------

def test_tokenize_yaml_handles_scanner_error_without_problem():
    exc = yaml.scanner.ScannerError("", 0, "", 0)
    exc.problem = None
    exc.problem_mark = None
    try:
        yaml.load("", CustomSafeLoader)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as caught_exc:
        assert caught_exc.problem is None


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_yaml_valid_yaml():
    content = "name: John\nage: 30"
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_empty_string():
    content = ""
    validator = Schema(fields={"name": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as error:
        assert error.text == "No content."
        assert error.code == "no_content"
        assert error.position == Position(line_no=1, column_no=1, char_index=0)

def test_validate_yaml_invalid_yaml_syntax():
    content = "name: John\n  age: 30"
    validator = Schema(fields={"name": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as error:
        assert error.code == "parse_error"
        assert error.position.char_index > 0

def test_validate_yaml_validation_error():
    content = "name: John"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]

def test_validate_yaml_with_positional_validation_error():
    content = "name: John\nage: thirty"
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(msg, 'start_position') for msg in messages)

def test_validate_yaml_bytes_input():
    content = b"name: John\nage: 30"
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_null_value():
    content = "name: null"
    validator = Schema(fields={"name": Field(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"name": None}

def test_validate_yaml_with_default_value():
    content = "{}"
    validator = Schema(fields={"name": Field(default="Default")})
    result = validate_yaml(content, validator)
    assert result == {"name": "Default"}

def test_validate_yaml_nested_structure():
    content = "user:\n  name: John\n  age: 30"
    validator = Schema(fields={"user": Schema(fields={"name": Field(), "age": Field()})})
    result = validate_yaml(content, validator)
    assert result == {"user": {"name": "John", "age": 30}}

def test_validate_yaml_with_read_only_field():
    content = "name: John"
    validator = Schema(fields={"name": Field(read_only=True)})
    result = validate_yaml(content, validator)
    assert result == {}

def test_validate_yaml_with_invalid_key_type():
    content = "123: value"
    validator = Schema(fields={"name": Field()})
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "invalid_key"


# LLM-generated content at query #22
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.base import Position
    from typesystem.exceptions import ParseError
    content = "invalid: yaml: :"
    try:
        token = typesystem.tokenize.tokenize_yaml.tokenize_yaml(content)
    except ParseError as exc:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml_raises_assertion_error_when_yaml_is_none():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field

    class MockField(Field):
        def validate(self, value):
            return value

    with patch.dict(sys.modules, {'yaml': None}):
        try:
            validate_yaml("test: value", MockField())
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #24
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        result = exc
    expected_text = "No content."
    expected_code = "no_content"
    expected_position = Position(column_no=1, line_no=1, char_index=0)
    assert result.text == expected_text
    assert result.code == expected_code
    assert result.position == expected_position

def test_tokenize_yaml_bytes_input():
    content = b"key: value"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

def test_tokenize_yaml_scalar_string():
    content = "hello"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

def test_tokenize_yaml_scalar_int():
    content = "42"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value == 42

def test_tokenize_yaml_scalar_float():
    content = "3.14"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

def test_tokenize_yaml_scalar_bool_true():
    content = "true"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value is True

def test_tokenize_yaml_scalar_bool_false():
    content = "false"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value is False

def test_tokenize_yaml_scalar_null():
    content = "null"
    result = tokenize_yaml(content)
    assert isinstance(result, ScalarToken)
    assert result.value is None

def test_tokenize_yaml_list():
    content = "- item1\n- item2"
    result = tokenize_yaml(content)
    assert isinstance(result, ListToken)
    assert result.value == ["item1", "item2"]

def test_tokenize_yaml_dict():
    content = "key: value"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

def test_tokenize_yaml_nested_structure():
    content = "key:\n  nested_key: nested_value"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert result.value == {"key": {"nested_key": "nested_value"}}

def test_tokenize_yaml_mixed_structure():
    content = "list:\n  - 1\n  - two\ndict:\n  key: value"
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    expected = {"list": [1, "two"], "dict": {"key": "value"}}
    assert result.value == expected

def test_tokenize_yaml_invalid_yaml():
    content = "key: [unclosed"
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        result = exc
    assert result.code == "parse_error"
    assert isinstance(result.position, Position)

def test_tokenize_yaml_token_positions():
    content = "key: value"
    result = tokenize_yaml(content)
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 10

def test_tokenize_yaml_token_string():
    content = "key: value"
    result = tokenize_yaml(content)
    assert result.string == "key: value"

def test_tokenize_yaml_lookup():
    content = "key: value"
    result = tokenize_yaml(content)
    child = result.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_yaml_lookup_key():
    content = "key: value"
    result = tokenize_yaml(content)
    key_token = result.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    sys.modules.pop('yaml', None)
    try:
        import typesystem.tokenize.tokenize_yaml
    except AssertionError as exc:
        assert str(exc) == "'pyyaml' must be installed."


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_yaml_raises_assertion_error_when_yaml_is_none():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field

    class MockField(Field):
        def validate(self, value):
            return value

    with patch.dict(sys.modules, {'yaml': None}):
        try:
            validate_yaml("test: value", MockField())
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  \t  ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)


def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value\nother: 123")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "other": 123}
    assert token.string == "key: value\nother: 123"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=9, char_index=22)


def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("list:\n  - item1\n  - item2\ndict:\n  key: val")
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"], "dict": {"key": "val"}}
    assert token.string == "list:\n  - item1\n  - item2\ndict:\n  key: val"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=5, column_no=9, char_index=41)


def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello: world")
    assert isinstance(token, DictToken)
    assert token.value == {"hello": "world"}
    assert token.string == "hello: world"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)


def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "did not find expected node content" in e.text or "did not find expected" in e.text
        assert isinstance(e.position, Position)


def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("a: 1\nb: 2")
    child = token.lookup(["b"])
    assert isinstance(child, ScalarToken)
    assert child.value == 2
    assert child.string == "2"
    assert child.start == Position(line_no=2, column_no=4, char_index=7)
    assert child.end == Position(line_no=2, column_no=4, char_index=7)


def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- x\n- y")
    child = token.lookup([1])
    assert isinstance(child, ScalarToken)
    assert child.value == "y"
    assert child.string == "y"
    assert child.start == Position(line_no=2, column_no=3, char_index=5)
    assert child.end == Position(line_no=2, column_no=3, char_index=5)


def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #2
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import typesystem.tokenize.tokenize_yaml
    import pytest
    class MockYAMLError(Exception):
        def __init__(self, problem, problem_mark):
            self.problem = problem
            self.problem_mark = problem_mark
    yaml = typesystem.tokenize.tokenize_yaml.yaml
    original_yaml = yaml
    try:
        class MockYAML:
            scanner = original_yaml.scanner
            parser = original_yaml.parser
            SafeLoader = original_yaml.SafeLoader
            resolver = original_yaml.resolver
            def load(self, content, loader):
                raise MockYAMLError(problem="test problem", problem_mark=None)
        typesystem.tokenize.tokenize_yaml.yaml = MockYAML()
        with pytest.raises(AssertionError):
            typesystem.tokenize.tokenize_yaml.tokenize_yaml("invalid yaml")
    finally:
        typesystem.tokenize.tokenize_yaml.yaml = original_yaml


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Position
    from typesystem.exceptions import ParseError

    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = None
            self.problem_mark = None

    content = "invalid: yaml: content"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_yaml_valid_simple_yaml():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    field = Integer()
    content = "42"
    result = validate_yaml(content, field)
    assert result == 42

def test_validate_yaml_invalid_simple_yaml():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    field = Integer()
    content = "not_an_integer"
    try:
        validate_yaml(content, field)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].start_position.char_index == 0

def test_validate_yaml_valid_object_yaml():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class MySchema(Schema):
        fields = {"name": String()}
    content = "name: John"
    result = validate_yaml(content, MySchema)
    assert result == {"name": "John"}

def test_validate_yaml_missing_required_field():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class MySchema(Schema):
        fields = {"name": String()}
    content = "age: 30"
    try:
        validate_yaml(content, MySchema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["name"]

def test_validate_yaml_invalid_field_type():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class MySchema(Schema):
        fields = {"age": Integer()}
    content = "age: not_an_integer"
    try:
        validate_yaml(content, MySchema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == ["age"]

def test_validate_yaml_empty_content():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    field = Integer()
    content = ""
    try:
        validate_yaml(content, field)
        assert False
    except ParseError as error:
        assert error.code == "no_content"

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    field = String()
    content = b"hello"
    result = validate_yaml(content, field)
    assert result == "hello"

def test_validate_yaml_invalid_yaml_syntax():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    field = String()
    content = "invalid: [unclosed"
    try:
        validate_yaml(content, field)
        assert False
    except ParseError as error:
        assert error.code == "parse_error"

def test_validate_yaml_with_default_value():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class MySchema(Schema):
        fields = {"name": String(default="Unknown")}
    content = "{}"
    result = validate_yaml(content, MySchema)
    assert result == {"name": "Unknown"}

def test_validate_yaml_with_allow_null():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class MySchema(Schema):
        fields = {"name": String(allow_null=True)}
    content = "name: null"
    result = validate_yaml(content, MySchema)
    assert result == {"name": None}

def test_validate_yaml_with_union_field():
    from typesystem.fields import Union, String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    field = String() | Integer()
    content = "123"
    result = validate_yaml(content, field)
    assert result == 123

def test_validate_yaml_with_nested_schema():
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class InnerSchema(Schema):
        fields = {"age": Integer()}
    class OuterSchema(Schema):
        fields = {"name": String(), "inner": InnerSchema}
    content = "name: John\ninner:\n  age: 30"
    result = validate_yaml(content, OuterSchema)
    assert result == {"name": "John", "inner": {"age": 30}}

def test_validate_yaml_with_list_token():
    from typesystem.fields import Array, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    field = Array(items=Integer())
    content = "- 1\n- 2\n- 3"
    result = validate_yaml(content, field)
    assert result == [1, 2, 3]

def test_validate_yaml_with_invalid_key_type():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class MySchema(Schema):
        fields = {"name": String()}
    content = "123: value"
    try:
        validate_yaml(content, MySchema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "invalid_key"
        assert messages[0].index == ["123"]


# LLM-generated content at query #5
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    content = b"invalid: \x81\x82"
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    import pytest
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = None
            self.problem_mark = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("invalid: [")
    except yaml.scanner.ScannerError as exc:
        if exc.problem is None:
            pytest.fail("exc.problem should not be None")


# LLM-generated content at query #7
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Position
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = None
            self.problem_mark = None
    try:
        tokenize_yaml("invalid: [")
    except Exception as e:
        pass


# LLM-generated content at query #8
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
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
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.string == "a:\n  b: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=11, char_index=13)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "parse_error" in exc.text.lower()

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=6, char_index=18)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\nother: *anchor")
    assert isinstance(token, DictToken)
    assert token.value == {"other": "value"}
    assert token.string == "&anchor value\nother: *anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=14, char_index=27)


# LLM-generated content at query #9
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
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
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)


def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)


def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - two")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, "two"]}
    assert token.string == "key:\n  - 1\n  - two"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=6, char_index=17)


def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index >= 0


# LLM-generated content at query #10
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
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

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=15)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("key: [")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 6
        assert exc.position.char_index == 5

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=19)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\nkey: *anchor")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "&anchor value\nkey: *anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=12, char_index=24)


# LLM-generated content at query #11
#--------------------------

def test_validate_yaml_does_not_raise_assertion_error_when_yaml_is_installed():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    dummy_field = Field()
    dummy_content = "key: value"
    with patch.dict(sys.modules, {'yaml': object()}):
        try:
            validate_yaml(dummy_content, dummy_field)
        except AssertionError:
            assert False, "AssertionError should not be raised when yaml is installed."


# LLM-generated content at query #12
#--------------------------

def test_validate_yaml_valid_yaml():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class StringField(Field):
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": StringField()})
    content = "name: John"
    result = validate_yaml(content, schema)
    assert result == {"name": "John"}

def test_validate_yaml_invalid_yaml_parse_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    class StringField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = "name: ["
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

def test_validate_yaml_empty_content():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    class StringField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = ""
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": StringField()})
    content = "name: 123"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "type"
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_yaml_required_field_missing():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class StringField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = "age: 30"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "required"
        assert message.index == ["name"]

def test_validate_yaml_nested_schema_validation():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class NumberField(Field):
        errors = {"type": "Must be a number."}
        def validate(self, value):
            if not isinstance(value, (int, float)):
                raise self.validation_error("type")
            return value
    inner_schema = Schema(fields={"city": StringField()})
    outer_schema = Schema(fields={"address": inner_schema, "age": NumberField()})
    content = "address:\n  city: 123\nage: thirty"
    try:
        validate_yaml(content, outer_schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "type" in codes
        for msg in messages:
            assert msg.start_position is not None
            assert msg.end_position is not None

def test_validate_yaml_bytes_input():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class StringField(Field):
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": StringField()})
    content = b"name: Alice"
    result = validate_yaml(content, schema)
    assert result == {"name": "Alice"}

def test_validate_yaml_allow_null_schema():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class StringField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()}, allow_null=True)
    content = "null"
    result = validate_yaml(content, schema)
    assert result is None

def test_validate_yaml_invalid_key_non_string():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    class StringField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = "123: value"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "invalid_key"
        assert message.index == [123]

def test_validate_yaml_default_value_applied():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class StringField(Field):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.default = "default_name"
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = "{}"
    result = validate_yaml(content, schema)
    assert result == {"name": "default_name"}

def test_validate_yaml_read_only_field_ignored():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    class StringField(Field):
        def __init__(self, **kwargs):
            kwargs["read_only"] = True
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    schema = Schema(fields={"name": StringField()})
    content = "name: John"
    result = validate_yaml(content, schema)
    assert result == {}


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = module.yaml
    module.yaml = None
    try:
        module.tokenize_yaml("test")
    except AssertionError as exc:
        assert str(exc) == "'pyyaml' must be installed."
    finally:
        module.yaml = original_yaml


# LLM-generated content at query #14
#--------------------------

def test_tokenize_yaml_empty_string():
    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    content = b""
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    content = "hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    content = "3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    content = "false"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    content = "- a\n- b"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=6)

def test_tokenize_yaml_dict():
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    content = "a:\n  b: [1, 2]"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.string == "a:\n  b: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=11, char_index=13)

def test_tokenize_yaml_bytes_input():
    content = b"test: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"test": "value"}
    assert token.string == "test: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_parse_error():
    content = "key: [unclosed"
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0


# LLM-generated content at query #15
#--------------------------

def test_validate_yaml_valid_content():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "name: John Doe\nage: 30"
    schema = Schema(fields={"name": String(), "age": String()})
    result = validate_yaml(content, schema)
    assert result == {"name": "John Doe", "age": "30"}

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    content = "name: John Doe\n  age: 30"
    schema = Schema(fields={"name": String(), "age": String()})
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

def test_validate_yaml_empty_string():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ParseError
    content = ""
    schema = Schema(fields={"name": String()})
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    content = "name: John Doe"
    schema = Schema(fields={"name": String(), "age": String()})
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["age"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = b"name: John Doe\nage: 30"
    schema = Schema(fields={"name": String(), "age": String()})
    result = validate_yaml(content, schema)
    assert result == {"name": "John Doe", "age": "30"}

def test_validate_yaml_nested_schema():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "person:\n  name: John\n  age: 30"
    person_schema = Schema(fields={"name": String(), "age": Integer()})
    schema = Schema(fields={"person": person_schema})
    result = validate_yaml(content, schema)
    assert result == {"person": {"name": "John", "age": 30}}

def test_validate_yaml_allow_null():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "name: null"
    schema = Schema(fields={"name": String(allow_null=True)})
    result = validate_yaml(content, schema)
    assert result == {"name": None}

def test_validate_yaml_invalid_key_type():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.exceptions import ValidationError
    content = "123: value"
    schema = Schema(fields={"name": String()})
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "invalid_key"
        assert msg.index == [123]

def test_validate_yaml_union_field():
    from typesystem.fields import String, Integer, Union
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "value: 42"
    union_field = String() | Integer()
    schema = Schema(fields={"value": union_field})
    result = validate_yaml(content, schema)
    assert result == {"value": 42}

def test_validate_yaml_default_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "{}"
    schema = Schema(fields={"name": String(default="Anonymous")})
    result = validate_yaml(content, schema)
    assert result == {"name": "Anonymous"}


# LLM-generated content at query #16
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_bytes_empty():
    try:
        tokenize_yaml(b"")
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

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=6)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested():
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=16)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "did not find expected node content" in exc.text

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #17
#--------------------------

def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    sys.modules['yaml'] = None
    try:
        module.tokenize_yaml("test")
    except AssertionError as e:
        result = str(e) == "'pyyaml' must be installed."
    finally:
        del sys.modules['yaml']
    assert result


# LLM-generated content at query #18
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Position
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self, problem, problem_mark):
            self.problem = problem
            self.problem_mark = problem_mark
    mock_error = MockScannerError(problem="test problem", problem_mark=None)
    yaml.load = lambda content, loader: (_ for _ in ()).throw(mock_error)
    try:
        tokenize_yaml("dummy content")
    except ParseError as e:
        pass


# LLM-generated content at query #19
#--------------------------

def test_validate_yaml_valid_content():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "42"
    result = validate_yaml(content, validator)
    assert result == 42

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml, ParseError
    validator = Integer()
    content = "invalid: ["
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"

def test_validate_yaml_empty_string():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml, ParseError
    validator = Integer()
    content = ""
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_whitespace_only():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml, ParseError
    validator = Integer()
    content = "   \n   "
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_with_schema():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"age": Integer(), "name": String()}
    validator = Schema(fields=fields)
    content = "age: 30\nname: John"
    result = validate_yaml(content, validator)
    assert result == {"age": 30, "name": "John"}

def test_validate_yaml_with_schema_validation_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.base import ValidationError
    fields = {"age": Integer(), "name": String()}
    validator = Schema(fields=fields)
    content = "age: thirty\nname: John"
    try:
        validate_yaml(content, validator)
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = String()
    content = b"hello"
    result = validate_yaml(content, validator)
    assert result == "hello"

def test_validate_yaml_complex_structure():
    from typesystem.schemas import Schema
    from typesystem.fields import Integer, String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"id": Integer(), "title": String()}
    validator = Schema(fields=fields)
    content = "id: 1\ntitle: Test"
    result = validate_yaml(content, validator)
    assert result == {"id": 1, "title": "Test"}

def test_validate_yaml_null_value():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer(allow_null=True)
    content = "null"
    result = validate_yaml(content, validator)
    assert result is None

def test_validate_yaml_invalid_yaml_syntax():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml, ParseError
    validator = Integer()
    content = "key: [unclosed"
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #20
#--------------------------

def test_tokenize_yaml_handles_scanner_error_without_problem():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    from unittest.mock import patch, Mock
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Position
    from typesystem.exceptions import ParseError
    exc = yaml.scanner.ScannerError("test")
    exc.problem = None
    exc.problem_mark = Mock()
    exc.problem_mark.index = 0
    with patch('yaml.load', side_effect=exc):
        try:
            typesystem.tokenize.tokenize_yaml.tokenize_yaml("test")
        except AssertionError as e:
            assert str(e) == "assert None is not None"


# LLM-generated content at query #21
#--------------------------

def test_validate_yaml_without_pyyaml_raises_assertion_error():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = module.yaml
    module.yaml = None
    try:
        module.validate_yaml(content="", validator=None)
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        module.yaml = original_yaml


# LLM-generated content at query #22
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

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

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("key:\n  - 1\n  - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, 2]}
    assert token.string == "key:\n  - 1\n  - 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=15)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=19)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\nkey: *anchor")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "&anchor value\nkey: *anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=13, char_index=26)


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    original_modules = sys.modules.copy()
    sys.modules['yaml'] = None
    try:
        import typesystem.tokenize.tokenize_yaml
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules.update(original_modules)


# LLM-generated content at query #24
#--------------------------

def test_tokenize_yaml_empty_string_raises_parse_error():
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    content = ""
    try:
        token = tokenize_yaml(content)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0


# LLM-generated content at query #25
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        result = exc
    assert result.code == "no_content"
    assert result.position.line_no == 1
    assert result.position.column_no == 1
    assert result.position.char_index == 0

def test_tokenize_yaml_empty_bytes():
    try:
        tokenize_yaml(b"")
    except ParseError as exc:
        result = exc
    assert result.code == "no_content"
    assert result.position.line_no == 1
    assert result.position.column_no == 1
    assert result.position.char_index == 0

def test_tokenize_yaml_scalar_string():
    result = tokenize_yaml("hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.string == "hello"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 5

def test_tokenize_yaml_scalar_int():
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.string == "42"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 2

def test_tokenize_yaml_scalar_float():
    result = tokenize_yaml("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.string == "3.14"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 4

def test_tokenize_yaml_scalar_bool_true():
    result = tokenize_yaml("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.string == "true"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 4

def test_tokenize_yaml_scalar_bool_false():
    result = tokenize_yaml("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.string == "false"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 5

def test_tokenize_yaml_scalar_null():
    result = tokenize_yaml("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.string == "null"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 4

def test_tokenize_yaml_list():
    result = tokenize_yaml("- a\n- b")
    assert isinstance(result, ListToken)
    assert result.value == ["a", "b"]
    assert result.string == "- a\n- b"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 2
    assert result.end.column_no == 3

def test_tokenize_yaml_dict():
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.string == "key: value"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 1
    assert result.end.column_no == 10

def test_tokenize_yaml_nested_structure():
    result = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(result, DictToken)
    assert result.value == {"a": {"b": [1, 2]}}
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 2
    assert result.end.column_no == 10

def test_tokenize_yaml_bytes_input():
    result = tokenize_yaml(b"hello")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.string == "hello"

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [unclosed")
    except ParseError as exc:
        result = exc
    assert result.code == "parse_error"
    assert result.position.line_no == 1
    assert result.position.column_no == 6

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("key: value")
    result = token.lookup(["key"])
    assert isinstance(result, ScalarToken)
    assert result.value == "value"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    result = token.lookup_key(["key"])
    assert isinstance(result, ScalarToken)
    assert result.value == "key"

def test_tokenize_yaml_multiline_string():
    result = tokenize_yaml("|\n  hello\n  world")
    assert isinstance(result, ScalarToken)
    assert result.value == "hello\nworld\n"
    assert result.start.line_no == 1
    assert result.start.column_no == 1
    assert result.end.line_no == 3
    assert result.end.column_no == 7

def test_tokenize_yaml_anchors_and_aliases():
    result = tokenize_yaml("&anchor value\n*anchor")
    assert isinstance(result, ListToken)
    assert result.value == ["value", "value"]

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n   ")
    except ParseError as exc:
        result = exc
    assert result.code == "no_content"

def test_tokenize_yaml_unicode():
    result = tokenize_yaml("café: naïve")
    assert isinstance(result, DictToken)
    assert result.value == {"café": "naïve"}
    assert result.string == "café: naïve"

def test_tokenize_yaml_quoted_scalar():
    result = tokenize_yaml("'42'")
    assert isinstance(result, ScalarToken)
    assert result.value == "42"
    assert result.string == "'42'"

def test_tokenize_yaml_flow_sequence():
    result = tokenize_yaml("[a, b, c]")
    assert isinstance(result, ListToken)
    assert result.value == ["a", "b", "c"]
    assert result.string == "[a, b, c]"

def test_tokenize_yaml_flow_mapping():
    result = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(result, DictToken)
    assert result.value == {"a": 1, "b": 2}
    assert result.string == "{a: 1, b: 2}"


# LLM-generated content at query #26
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n\t  ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"
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
    token = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.string == "a:\n  b: [1, 2]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=11, char_index=14)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml(": invalid")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0

def test_tokenize_yaml_multiline_scalar():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=18)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\n*anchor")
    assert isinstance(token, ListToken)
    assert token.value == ["value", "value"]
    assert token.string == "&anchor value\n*anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=19)

def test_tokenize_yaml_complex_mapping():
    token = tokenize_yaml("? complex key\n: value")
    assert isinstance(token, DictToken)
    assert list(token.value.keys())[0] == "complex key"
    assert list(token.value.values())[0] == "value"
    assert token.string == "? complex key\n: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=20)

def test_tokenize_yaml_quoted_scalars():
    token = tokenize_yaml("'single'\n\"double\"")
    assert isinstance(token, ListToken)
    assert token.value == ["single", "double"]
    assert token.string == "'single'\n\"double\""
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=8, char_index=16)


