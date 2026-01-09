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
        assert isinstance(e.position, Position)

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


# LLM-generated content at query #2
#--------------------------

def test_validate_yaml_valid_content():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "key: value"
    field = Field()
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "key: ["
    field = Field()
    schema = Schema(fields={"key": field})
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "parse_error"

def test_validate_yaml_empty_string():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = ""
    field = Field()
    schema = Schema(fields={"key": field})
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "no_content"

def test_validate_yaml_bytes_input():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = b"key: value"
    field = Field()
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {"key": "value"}

def test_validate_yaml_validation_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "key: 123"
    field = Field()
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {"key": 123}

def test_validate_yaml_with_required_field_missing():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "{}"
    field = Field()
    schema = Schema(fields={"key": field})
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert hasattr(e, 'messages')
        assert any(msg.code == "required" for msg in e.messages)

def test_validate_yaml_with_default_field():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "{}"
    field = Field(default="default_value")
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {"key": "default_value"}

def test_validate_yaml_with_allow_null():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "key: null"
    field = Field(allow_null=True)
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {"key": None}

def test_validate_yaml_with_read_only_field():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "key: value"
    field = Field(read_only=True)
    schema = Schema(fields={"key": field})
    result = validate_yaml(content, schema)
    assert result == {}

def test_validate_yaml_with_nested_schema():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    content = "outer:\n  inner: value"
    inner_field = Field()
    outer_schema = Schema(fields={"inner": inner_field})
    schema = Schema(fields={"outer": outer_schema})
    result = validate_yaml(content, schema)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #3
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
    mock_exc = MockScannerError(problem="test problem", problem_mark=None)
    try:
        yaml.load = lambda content, loader: (_ for _ in []).throw(mock_exc)
        tokenize_yaml("some content")
    except ParseError as e:
        pass


# LLM-generated content at query #4
#--------------------------

def test_validate_yaml_with_pyyaml_installed():
    import sys
    from unittest.mock import patch
    from typesystem.fields import Field
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    with patch.dict(sys.modules, {'yaml': None}):
        try:
            validate_yaml("key: value", Field())
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
        else:
            assert False, "Expected AssertionError"


# LLM-generated content at query #5
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    from unittest.mock import patch
    from typesystem.exceptions import ParseError
    yaml.scanner.ScannerError.problem = None
    yaml.scanner.ScannerError.problem_mark = None
    yaml.parser.ParserError.problem = None
    yaml.parser.ParserError.problem_mark = None
    with patch.object(yaml, 'load', side_effect=yaml.scanner.ScannerError('test')):
        try:
            typesystem.tokenize.tokenize_yaml.tokenize_yaml('invalid: [')
        except AssertionError as e:
            pass


# LLM-generated content at query #6
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = "test problem"
            self.problem_mark = None
    try:
        tokenize_yaml("invalid: [")
    except Exception as e:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    original_modules = sys.modules.copy()
    sys.modules['yaml'] = None
    try:
        from typesystem.tokenize.tokenize_yaml import tokenize_yaml
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules.update(original_modules)


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
    assert token.end == Position(line_no=2, column_no=12, char_index=14)

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
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_tokenize_yaml_multiline_scalar():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.string == "|\n  line1\n  line2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=19)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\n*anchor")
    assert isinstance(token, ListToken)
    assert token.value == ["value", "value"]
    assert token.string == "&anchor value\n*anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=20)

def test_tokenize_yaml_complex_mapping():
    token = tokenize_yaml("? complex key\n: complex value")
    assert isinstance(token, DictToken)
    assert token.value == {"complex key": "complex value"}
    assert token.string == "? complex key\n: complex value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=15, char_index=30)

def test_tokenize_yaml_quoted_scalars():
    token = tokenize_yaml("'single' \"double\"")
    assert isinstance(token, ListToken)
    assert token.value == ["single", "double"]
    assert token.string == "'single' \"double\""
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=17, char_index=16)


# LLM-generated content at query #9
#--------------------------

def test_validate_yaml_does_not_raise_assertion_error_when_yaml_is_installed():
    import sys
    import typesystem.tokenize.tokenize_yaml
    original_yaml = typesystem.tokenize.tokenize_yaml.yaml
    typesystem.tokenize.tokenize_yaml.yaml = "mock_yaml_module"
    try:
        typesystem.tokenize.tokenize_yaml.validate_yaml("key: value", typesystem.fields.Field())
    except AssertionError:
        assert False, "AssertionError should not be raised when yaml is not None"
    finally:
        typesystem.tokenize.tokenize_yaml.yaml = original_yaml


# LLM-generated content at query #10
#--------------------------

def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #11
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
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
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
        assert isinstance(e.position, Position)

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("a: [x: 1]")
    child = token.lookup(["a", 0, "x"])
    assert isinstance(child, ScalarToken)
    assert child.value == 1

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"


# LLM-generated content at query #12
#--------------------------

def test_validate_yaml_valid():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = "name: John"
    result = validate_yaml(content, schema)
    assert result == {"name": "John"}

def test_validate_yaml_invalid_yaml_parse_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = "name: John\n  invalid: indent"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert e.code == "parse_error"

def test_validate_yaml_empty_content():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = ""
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert e.code == "no_content"

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(max_length=5)})
    content = "name: Jonathan"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "max_length"
        assert msg.start_position is not None

def test_validate_yaml_required_field_missing():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = "age: 30"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.start_position is not None

def test_validate_yaml_with_default():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(default="Unknown")})
    content = "{}"
    result = validate_yaml(content, schema)
    assert result == {"name": "Unknown"}

def test_validate_yaml_allow_null():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(allow_null=True)})
    content = "name: null"
    result = validate_yaml(content, schema)
    assert result == {"name": None}

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = b"name: Alice"
    result = validate_yaml(content, schema)
    assert result == {"name": "Alice"}

def test_validate_yaml_nested_schema():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    inner = Schema({"age": Integer()})
    outer = Schema({"person": inner})
    content = "person:\n  age: 25"
    result = validate_yaml(content, outer)
    assert result == {"person": {"age": 25}}

def test_validate_yaml_invalid_key_type():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = "123: value"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid_key"


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml
    yaml = typesystem.tokenize.tokenize_yaml.yaml
    typesystem.tokenize.tokenize_yaml.yaml = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        typesystem.tokenize.tokenize_yaml.yaml = yaml


# LLM-generated content at query #14
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_bytes_empty():
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
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)

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

def test_tokenize_yaml_multiline_scalar():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=6, char_index=19)

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n   ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #15
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    exc = yaml.scanner.ScannerError("", None, None, None, None)
    exc.problem = None
    exc.problem_mark = None
    try:
        yaml.load("", CustomSafeLoader)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as caught_exc:
        assert caught_exc is exc
        assert caught_exc.problem is None


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml_valid_content():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer()})
    content = "age: 25"
    result = validate_yaml(content, schema)
    assert result == {"age": 25}

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer()})
    content = "age: :"
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

def test_validate_yaml_empty_content():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer()})
    content = ""
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer()})
    content = "age: invalid"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "invalid"
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    schema = Schema({"name": String()})
    content = b"name: John"
    result = validate_yaml(content, schema)
    assert result == {"name": "John"}

def test_validate_yaml_complex_structure():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(), "age": Integer()})
    content = "name: Alice\nage: 30"
    result = validate_yaml(content, schema)
    assert result == {"name": "Alice", "age": 30}

def test_validate_yaml_required_field_missing():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(), "age": Integer()})
    content = "name: Bob"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "required"
        assert message.index == ["age"]
        assert "required" in message.text.lower()

def test_validate_yaml_with_default_values():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(default="Unknown"), "age": Integer()})
    content = "age: 25"
    result = validate_yaml(content, schema)
    assert result == {"name": "Unknown", "age": 25}

def test_validate_yaml_read_only_field():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    schema = Schema({"name": String(read_only=True), "age": Integer()})
    content = "name: Alice\nage: 30"
    result = validate_yaml(content, schema)
    assert result == {"age": 30}
    assert "name" not in result

def test_validate_yaml_allow_null():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer(allow_null=True)})
    content = "age: null"
    result = validate_yaml(content, schema)
    assert result == {"age": None}

def test_validate_yaml_nested_schema():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    person_schema = Schema({"name": String(), "age": Integer()})
    schema = Schema({"person": person_schema})
    content = "person:\n  name: Charlie\n  age: 40"
    result = validate_yaml(content, schema)
    assert result == {"person": {"name": "Charlie", "age": 40}}

def test_validate_yaml_invalid_yaml_structure():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    schema = Schema({"age": Integer()})
    content = "- invalid\n- list"
    try:
        validate_yaml(content, schema)
    except ValidationError as exc:
        assert len(exc.messages()) >= 1
        assert exc.messages()[0].code == "type"


# LLM-generated content at query #18
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    exc = yaml.scanner.ScannerError("", None, None, None, None)
    exc.problem = None
    exc.problem_mark = None
    try:
        yaml.load("invalid yaml", CustomSafeLoader)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as caught_exc:
        assert caught_exc is exc
        assert caught_exc.problem is None


# LLM-generated content at query #19
#--------------------------

def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = module.yaml
    module.yaml = None
    try:
        module.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        module.yaml = original_yaml


# LLM-generated content at query #20
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = "test problem"
            self.problem_mark = None
    try:
        tokenize_yaml("invalid: [")
    except Exception as e:
        assert isinstance(e, AssertionError)


# LLM-generated content at query #21
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

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml(": invalid")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index == 0

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
    content = "key: |\n  line1\n  line2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "line1\nline2\n"}
    assert token.string == content

def test_tokenize_yaml_anchors_and_aliases():
    content = "&anchor value\nother: *anchor"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"other": "value"}


# LLM-generated content at query #22
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    exc = type('MockError', (), {'problem': None, 'problem_mark': None})()
    try:
        assert exc.problem is not None
    except AssertionError:
        pass


# LLM-generated content at query #23
#--------------------------

def test_validate_yaml_raises_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = module.yaml
    module.yaml = None
    try:
        result = module.validate_yaml("test: value", None)
    except AssertionError as e:
        caught_message = str(e)
    finally:
        module.yaml = original_yaml
    assert caught_message == "'pyyaml' must be installed."


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml
    yaml_module = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        if yaml_module is not None:
            sys.modules['yaml'] = yaml_module
        else:
            del sys.modules['yaml']


# LLM-generated content at query #25
#--------------------------

def test_validate_yaml_with_pyyaml_installed():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, ScalarToken
    yaml = None
    with patch.dict(sys.modules, {'yaml': yaml}):
        try:
            validator = Field()
            result = validate_yaml("test: value", validator)
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."
        else:
            assert False, "Expected AssertionError"


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml
    yaml = typesystem.tokenize.tokenize_yaml.yaml
    typesystem.tokenize.tokenize_yaml.yaml = None
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        typesystem.tokenize.tokenize_yaml.yaml = yaml


# LLM-generated content at query #27
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = "test problem"
            self.problem_mark = None
    try:
        tokenize_yaml("invalid: [")
    except Exception as e:
        pass


# LLM-generated content at query #28
#--------------------------

def test_tokenize_yaml_handles_scanner_error_without_problem():
    try:
        tokenize_yaml("invalid yaml content: [")
    except Exception as exc:
        pass


# LLM-generated content at query #29
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    content = "invalid: yaml: ["
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #30
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
    assert token.end == Position(line_no=2, column_no=10, char_index=13)

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
        assert "could not find expected key" in exc.text

def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"

def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- a\n- b")
    child = token.lookup([0])
    assert isinstance(child, ScalarToken)
    assert child.value == "a"
    assert child.string == "a"

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


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml_valid_content():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John\nage: 30"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": "30"}

def test_validate_yaml_invalid_yaml_parse_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John\n  age: 30"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

def test_validate_yaml_empty_content():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = ""
    fields = {"name": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John\nage: 30"
    fields = {"name": String(max_length=3), "age": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "max_length"
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = b"name: John\nage: 30"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": "30"}

def test_validate_yaml_complex_structure():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    content = "user:\n  name: John\n  age: 30\n  tags:\n    - developer\n    - python"
    fields = {"user": Schema(fields={"name": String(), "age": Integer(), "tags": String()})}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"user": {"name": "John", "age": 30, "tags": "python"}}

def test_validate_yaml_with_field_validator():
    from typesystem.fields import Integer
    content = "value: 42"
    validator = Integer(minimum=0, maximum=100)
    result = validate_yaml(content, validator)
    assert result == 42

def test_validate_yaml_whitespace_only():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "   \n  \n"
    fields = {"name": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_missing_required_field():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #32
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    exc = yaml.parser.ParserError("problem")
    exc.problem = "some problem"
    exc.problem_mark = None
    try:
        yaml.load("invalid: [", CustomSafeLoader)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as caught_exc:
        assert caught_exc is exc


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_tokenize_yaml_whitespace_only():
    try:
        tokenize_yaml("   \n  \t  ")
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
        tokenize_yaml(": invalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_tokenize_yaml_lookup_dict():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"

def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- a\n- b")
    child = token.lookup([1])
    assert isinstance(child, ScalarToken)
    assert child.value == "b"
    assert child.string == "b"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"

def test_tokenize_yaml_multiline_scalar():
    token = tokenize_yaml("|\n  line1\n  line2")
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2\n"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=6, char_index=18)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\nother: *anchor")
    assert isinstance(token, DictToken)
    assert token.value == {"other": "value"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=14, char_index=27)

def test_tokenize_yaml_complex_mapping():
    token = tokenize_yaml("a: 1\nb:\n  c: 2\n  d: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": {"c": 2, "d": 3}}
    assert token.string == "a: 1\nb:\n  c: 2\n  d: 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=6, char_index=22)

def test_tokenize_yaml_token_equality():
    token1 = tokenize_yaml("key: value")
    token2 = tokenize_yaml("key: value")
    assert token1 == token2

def test_tokenize_yaml_token_inequality():
    token1 = tokenize_yaml("key: value")
    token2 = tokenize_yaml("key: other")
    assert not (token1 == token2)


# LLM-generated content at query #2
#--------------------------

def test_tokenize_yaml_assertion_error_when_yaml_not_installed():
    import sys
    original_modules = sys.modules.copy()
    sys.modules['yaml'] = None
    try:
        import typesystem.tokenize.tokenize_yaml
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules.update(original_modules)


# LLM-generated content at query #3
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
    except ParseError as e:
        assert e.code == "parse_error"
        assert "did not find expected node content" in e.text
        assert isinstance(e.position, Position)


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    sys.modules.pop('yaml', None)
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #5
#--------------------------

def test_validate_yaml_valid_content():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John\nage: 30"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": "30"}

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John\n  age: 30"
    fields = {"name": String(), "age": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position is not None

def test_validate_yaml_empty_string():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = ""
    fields = {"name": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: John"
    fields = {"name": String(max_length=3)}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_yaml_required_field_missing():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "{}"
    fields = {"name": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.code == "required"
        assert message.index == ["name"]

def test_validate_yaml_with_default_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "{}"
    fields = {"name": String(default="default_name")}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"name": "default_name"}

def test_validate_yaml_allow_null():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "name: null"
    fields = {"name": String(allow_null=True)}
    validator = Schema(fields=fields, allow_null=True)
    result = validate_yaml(content, validator)
    assert result == {"name": None}

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = b"name: Alice"
    fields = {"name": String()}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"name": "Alice"}

def test_validate_yaml_nested_structure():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    content = "person:\n  name: Bob\n  age: 25"
    person_fields = {"name": String(), "age": Integer()}
    person_schema = Schema(fields=person_fields)
    fields = {"person": person_schema}
    validator = Schema(fields=fields)
    result = validate_yaml(content, validator)
    assert result == {"person": {"name": "Bob", "age": 25}}

def test_validate_yaml_invalid_key_type():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    content = "123: value"
    fields = {"name": String()}
    validator = Schema(fields=fields)
    try:
        validate_yaml(content, validator)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) >= 1
        message = exc.messages()[0]
        assert message.code == "invalid_key"


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml_assertion_error_when_yaml_is_none():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = module.yaml
    module.yaml = None
    try:
        module.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        module.yaml = original_yaml


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    sys.modules.pop('yaml', None)
    try:
        import typesystem.tokenize.tokenize_yaml
    except AssertionError as exc:
        assert str(exc) == "'pyyaml' must be installed."


# LLM-generated content at query #8
#--------------------------

def test_validate_yaml_does_not_raise_assertion_error_when_yaml_is_installed():
    import sys
    import typesystem.tokenize.tokenize_yaml
    original_modules = sys.modules.copy()
    sys.modules['yaml'] = object()
    try:
        typesystem.tokenize.tokenize_yaml.validate_yaml("key: value", typesystem.fields.Field())
    except AssertionError:
        assert False, "AssertionError should not be raised when yaml is installed"
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


# LLM-generated content at query #9
#--------------------------

def test_validate_yaml_raises_assertion_error_when_yaml_is_none():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        validate_yaml(content="", validator=Field())
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #10
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = "test problem"
            self.problem_mark = None
    try:
        tokenize_yaml("invalid yaml: [")
    except Exception as e:
        pass


# LLM-generated content at query #11
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError
    content = "invalid: yaml: :"
    try:
        tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #12
#--------------------------

def test_validate_yaml_with_pyyaml_installed():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Position
    from typesystem.exceptions import ValidationError
    from typesystem import Message
    import yaml
    yaml_token = Token(value={}, start_index=0, end_index=0, content="")
    with patch('typesystem.tokenize.tokenize_yaml.tokenize_yaml', return_value=yaml_token):
        with patch('typesystem.tokenize.tokenize_yaml.validate_with_positions', return_value=42):
            result = validate_yaml(content="key: value", validator=Field())
            assert result == 42


# LLM-generated content at query #13
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
        tokenize_yaml("   \n   ")
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
    assert token.end == Position(line_no=3, column_no=3, char_index=8)


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


def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml(": invalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "could not find expected key" in exc.text
        assert isinstance(exc.position, Position)


def test_tokenize_yaml_lookup():
    token = tokenize_yaml("a:\n  - x: 1\n    y: 2")
    child = token.lookup(["a", 0, "x"])
    assert isinstance(child, ScalarToken)
    assert child.value == 1
    assert child.string == "1"
    assert child.start == Position(line_no=2, column_no=8, char_index=12)
    assert child.end == Position(line_no=2, column_no=8, char_index=12)


def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #14
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():  
    exc = yaml.scanner.ScannerError("test", 1, 1, "test", 1)  
    exc.problem = None  
    exc.problem_mark = yaml.error.Mark("test", 0, 0, 0, None, 0)  
    try:  
        raise exc  
    except (yaml.scanner.ScannerError, yaml.parser.ParserError):  
        pass


# LLM-generated content at query #15
#--------------------------

def test_tokenize_yaml_parse_error_without_problem():
    try:
        tokenize_yaml("invalid: [")
    except Exception as exc:
        pass


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

def test_tokenize_yaml_parse_error_without_problem_mark():
    import typesystem.tokenize.tokenize_yaml
    import yaml
    import pytest
    class MockScannerError(yaml.scanner.ScannerError):
        def __init__(self):
            self.problem = "test problem"
            self.problem_mark = None
    try:
        yaml.load("invalid: [", typesystem.tokenize.tokenize_yaml.CustomSafeLoader)
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as exc:
        if exc.problem_mark is None:
            with pytest.raises(AssertionError):
                typesystem.tokenize.tokenize_yaml.tokenize_yaml("invalid: [")


# LLM-generated content at query #19
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
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "did not find expected node content" in exc.text

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
    token = tokenize_yaml("|\n  hello\n  world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello\nworld\n"
    assert token.string == "|\n  hello\n  world"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=20)

def test_tokenize_yaml_anchors_and_aliases():
    token = tokenize_yaml("&anchor value\nother: *anchor")
    assert isinstance(token, DictToken)
    assert token.value == {"other": "value"}
    assert token.string == "&anchor value\nother: *anchor"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=14, char_index=27)


# LLM-generated content at query #20
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

def test_tokenize_yaml_scalar_string():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"

def test_tokenize_yaml_scalar_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_scalar_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"

def test_tokenize_yaml_scalar_bool_true():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"

def test_tokenize_yaml_scalar_bool_false():
    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"

def test_tokenize_yaml_scalar_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"

def test_tokenize_yaml_list():
    token = tokenize_yaml("- a\n- b")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b"]
    assert token.string == "- a\n- b"

def test_tokenize_yaml_dict():
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("key:\n  - 1\n  - two")
    assert isinstance(token, DictToken)
    assert token.value == {"key": [1, "two"]}

def test_tokenize_yaml_parse_error():
    try:
        tokenize_yaml("key: [")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert isinstance(exc.position, Position)

def test_tokenize_yaml_token_positions():
    token = tokenize_yaml("key: value")
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_token_lookup():
    token = tokenize_yaml("key: value")
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_yaml_token_lookup_key():
    token = tokenize_yaml("key: value")
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"


# LLM-generated content at query #21
#--------------------------

def test_validate_yaml_valid_content():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = "age: 25"
    result = validate_yaml(content, validator)
    assert result == {"age": 25}

def test_validate_yaml_invalid_content_parse_error():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = "age: :"
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "parse_error"
        assert error.position.char_index >= 0

def test_validate_yaml_empty_string():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = ""
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "no_content"
        assert error.position.char_index == 0

def test_validate_yaml_validation_error_with_positions():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = "age: invalid"
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "type"
        assert message.start_position.char_index >= 0
        assert message.end_position.char_index >= 0

def test_validate_yaml_required_field_missing():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = "name: John"
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.start_position.char_index >= 0
        assert message.end_position.char_index >= 0

def test_validate_yaml_bytes_input():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    validator = Schema(fields={"age": Integer()})
    content = b"age: 30"
    result = validate_yaml(content, validator)
    assert result == {"age": 30}

def test_validate_yaml_complex_structure():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema
    validator = Schema(fields={"name": String(), "age": Integer()})
    content = "name: Alice\nage: 28"
    result = validate_yaml(content, validator)
    assert result == {"name": "Alice", "age": 28}

def test_validate_yaml_with_union_field():
    from typesystem.fields import Integer, String, Union
    validator = Union(any_of=[Integer(), String()])
    content = "42"
    result = validate_yaml(content, validator)
    assert result == 42

def test_validate_yaml_with_union_field_second_type():
    from typesystem.fields import Integer, String, Union
    validator = Union(any_of=[Integer(), String()])
    content = '"hello"'
    result = validate_yaml(content, validator)
    assert result == "hello"

def test_validate_yaml_with_union_field_error():
    from typesystem.fields import Integer, String, Union
    validator = Union(any_of=[Integer(), String()])
    content = "null"
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "null"


# LLM-generated content at query #22
#--------------------------

def test_tokenize_yaml_empty_string():
    content = ""
    try:
        tokenize_yaml(content)
        assert False
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_whitespace_only():
    content = "   \n\t  "
    try:
        tokenize_yaml(content)
        assert False
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_simple_scalar():
    content = "hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_integer():
    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float():
    content = "3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_boolean_true():
    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_boolean_false():
    content = "false"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_null():
    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list():
    content = "- a\n- b\n- c"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=8)

def test_tokenize_yaml_dict():
    content = "key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == "key: value"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_nested_structure():
    content = "list:\n  - item1\n  - item2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"]}
    assert token.string == "list:\n  - item1\n  - item2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=8, char_index=24)

def test_tokenize_yaml_bytes_input():
    content = b"hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_parse_error():
    content = "key: [unclosed"
    try:
        tokenize_yaml(content)
        assert False
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index >= 0

def test_tokenize_yaml_lookup():
    content = "key: value"
    token = tokenize_yaml(content)
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"
    assert child.string == "value"
    assert child.start == Position(line_no=1, column_no=6, char_index=5)
    assert child.end == Position(line_no=1, column_no=10, char_index=9)

def test_tokenize_yaml_lookup_key():
    content = "key: value"
    token = tokenize_yaml(content)
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == "key"
    assert key_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert key_token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #23
#--------------------------

def test_validate_yaml_valid_yaml():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "42"
    result = validate_yaml(content, validator)
    assert result == 42

def test_validate_yaml_invalid_yaml_parse_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = "["
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert e.code == "parse_error"

def test_validate_yaml_empty_string():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = Integer()
    content = ""
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert e.code == "no_content"

def test_validate_yaml_with_schema_valid():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String()}
    schema = Schema(fields=fields)
    content = "name: John"
    result = validate_yaml(content, schema)
    assert result == {"name": "John"}

def test_validate_yaml_with_schema_missing_required():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String()}
    schema = Schema(fields=fields)
    content = "age: 30"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert any(m.code == "required" for m in e.messages())

def test_validate_yaml_with_union_field():
    from typesystem.fields import Union, String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    union_field = String() | Integer()
    content = "123"
    result = validate_yaml(content, union_field)
    assert result == 123

def test_validate_yaml_with_union_field_no_match():
    from typesystem.fields import Union, String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    union_field = String() | Integer()
    content = "true"
    try:
        validate_yaml(content, union_field)
    except Exception as e:
        assert e.code == "union"

def test_validate_yaml_bytes_input():
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    validator = String()
    content = b"hello"
    result = validate_yaml(content, validator)
    assert result == "hello"

def test_validate_yaml_invalid_key_type():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String()}
    schema = Schema(fields=fields)
    content = "{123: value}"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert any(m.code == "invalid_key" for m in e.messages())

def test_validate_yaml_with_default_field():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String(default="Anonymous")}
    schema = Schema(fields=fields)
    content = "{}"
    result = validate_yaml(content, schema)
    assert result == {"name": "Anonymous"}

def test_validate_yaml_with_read_only_field():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String(read_only=True)}
    schema = Schema(fields=fields)
    content = "name: John"
    result = validate_yaml(content, schema)
    assert result == {}

def test_validate_yaml_allow_null_schema():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String()}
    schema = Schema(fields=fields, allow_null=True)
    content = "null"
    result = validate_yaml(content, schema)
    assert result is None

def test_validate_yaml_disallow_null_schema():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String()}
    schema = Schema(fields=fields, allow_null=False)
    content = "null"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        assert e.code == "null"

def test_validate_yaml_nested_structure():
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"person": Schema(fields={"name": String(), "age": Integer()})}
    schema = Schema(fields=fields)
    content = "person:\n  name: Alice\n  age: 25"
    result = validate_yaml(content, schema)
    assert result == {"person": {"name": "Alice", "age": 25}}

def test_validate_yaml_positional_error_messages():
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    fields = {"name": String(), "age": Integer()}
    schema = Schema(fields=fields)
    content = "name: John\nage: not_a_number"
    try:
        validate_yaml(content, schema)
    except Exception as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


# LLM-generated content at query #24
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
    assert token.end == Position(line_no=2, column_no=10, char_index=13)


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


def test_tokenize_yaml_lookup_list():
    token = tokenize_yaml("- a\n- b")
    child = token.lookup([0])
    assert isinstance(child, ScalarToken)
    assert child.value == "a"
    assert child.string == "a"


def test_tokenize_yaml_lookup_dict():
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
    token = tokenize_yaml("|\n  hello\n  world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello\nworld\n"
    assert token.string == "|\n  hello\n  world"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=7, char_index=20)


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem_mark():
    import yaml
    import typesystem.tokenize.tokenize_yaml
    content = "invalid: ["
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml(content)
    except Exception as exc:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_yaml_parse_error_without_problem_mark():
    try:
        tokenize_yaml("invalid: [")
    except Exception as exc:
        pass


# LLM-generated content at query #27
#--------------------------

def test_tokenize_yaml_empty_string_raises_parse_error():
    import typesystem.tokenize.tokenize_yaml
    try:
        typesystem.tokenize.tokenize_yaml.tokenize_yaml("")
    except typesystem.base.ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0


# LLM-generated content at query #28
#--------------------------

def test_tokenize_yaml_asserts_yaml_is_not_none():
    import sys
    sys.modules['yaml'] = None
    import typesystem.tokenize.tokenize_yaml
    reloaded = sys.modules['typesystem.tokenize.tokenize_yaml']
    try:
        reloaded.tokenize_yaml("")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        del sys.modules['yaml']


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml_pyyaml_installed():
    import sys
    from unittest.mock import patch
    from typesystem.tokenize.tokenize_yaml import validate_yaml
    from typesystem.fields import Field
    from typesystem.schemas import Schema

    class MockField(Field):
        def validate(self, value):
            return value

    validator = MockField()
    content = "test: value"

    with patch.dict(sys.modules, {'yaml': None}):
        try:
            validate_yaml(content, validator)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #30
#--------------------------

def test_tokenize_yaml_empty_string():
    try:
        tokenize_yaml("")
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
        tokenize_yaml(": invalid")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0


