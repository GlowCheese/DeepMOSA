####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_yaml_empty_string_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar_token():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.string == "hello"
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list_token():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.string == "[1, 2, 3]"
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict_token():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.string == "{a: 1, b: 2}"
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - c\n    - d")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": ["c", "d"]}}
    assert token.lookup([0]).value == {"b": ["c", "d"]}
    assert token.lookup([0, 0]).value == ["c", "d"]
    assert token.lookup([0, 0, 0]).value == "c"
    assert token.lookup([0, 0, 1]).value == "d"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("a:\n  b: c")
    key_token = token.lookup_key([0, 0])
    assert key_token.value == "b"
    assert key_token.string == "b"

def test_tokenize_yaml_invalid_yaml_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a:\n  b: c\n  d: e\n    f: g")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("a:\n  b: c")
    assert token.value == {"a": {"b": "c"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=9)

def test_tokenize_yaml_special_types():
    token = tokenize_yaml("int: 1\nfloat: 1.0\nbool: true\nnull: null")
    assert token.value == {"int": 1, "float": 1.0, "bool": True, "null": None}
    assert isinstance(token.lookup([0]), ScalarToken)
    assert isinstance(token.lookup([1]), ScalarToken)
    assert isinstance(token.lookup([2]), ScalarToken)
    assert isinstance(token.lookup([3]), ScalarToken)


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{a: 1, b: 2,}")
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

def test_tokenize_yaml_multiline_content():
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=6)

def test_tokenize_yaml_special_types():
    token = tokenize_yaml("int: 1\nfloat: 1.5\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 1, "float": 1.5, "bool": True, "null": None}
    assert isinstance(token.lookup(["int"]), ScalarToken)
    assert isinstance(token.lookup(["float"]), ScalarToken)
    assert isinstance(token.lookup(["bool"]), ScalarToken)
    assert isinstance(token.lookup(["null"]), ScalarToken)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml_with_valid_content_and_validator():
    content = "key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == "value"

def test_validate_yaml_with_invalid_yaml_content():
    content = "key: [value"
    validator = Field()
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Field()
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_with_null_content_and_allow_null():
    content = "null"
    validator = Field(allow_null=True)
    result = validate_yaml(content, validator)
    assert result is None

def test_validate_yaml_with_invalid_validation():
    content = "key: value"
    validator = Field()
    validator.validate = lambda x: 1 / 0
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages is not None

def test_validate_yaml_with_schema_validator():
    content = "key: value"
    validator = Schema(fields={"key": Field()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_schema_validator_and_missing_required_field():
    content = "other_key: value"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)

def test_validate_yaml_with_schema_validator_and_invalid_key_type():
    content = "123: value"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    import sys
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module

    # Simulate the absence of pyyaml by setting yaml to None
    yaml = None
    sys.modules['yaml'] = None

    # Reload the module to ensure the yaml import is re-evaluated
    import importlib
    importlib.reload(tokenize_yaml_module)

    # Verify the assertion fails when yaml is None
    try:
        tokenize_yaml_module.tokenize_yaml("test")
        assert False, "Expected AssertionError was not raised"
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup_key(["a"]).value == "a"
    assert isinstance(token.lookup(["a"]), ListToken)
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup_key(["b"]).value == "b"
    assert isinstance(token.lookup(["b"]), DictToken)
    assert token.lookup_key(["b", "c"]).value == "c"
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: c")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=7)
    assert token.lookup_key(["a"]).value == "a"
    assert isinstance(token.lookup(["a"]), DictToken)
    assert token.lookup_key(["a", "b"]).value == "b"
    assert token.lookup(["a", "b"]).value == "c"

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "test: value"
    validator = Field()
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        validate_yaml(content, validator)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_False():
    exc = yaml.scanner.ScannerError("test", None, None, None, None, None)
    assert exc.problem is None


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("{a: {b: [1, 2, 3]}}")
    assert token.lookup(["a", "b", 1]).value == 2

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("{a: {b: 1}}")
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=10)

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a: [1, 2, 3")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: c")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=7)
    assert token.lookup(["a", "b"]).value == "c"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [b")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None
    assert exc.problem_mark is None


# LLM-generated content at query #11
#--------------------------

```python
def test_line_84_predicate_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar_token():
    token = tokenize_yaml("test")
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == "test"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_list_token():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict_token():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a:\n  b: [1\n    2]")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"test")
    assert isinstance(token, ScalarToken)
    assert token.value == "test"

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("a:\n  b: c")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c"}}
    assert token.lookup(["a", "b"]).string == "c"


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    yaml = None
    assert yaml is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test problem", problem_mark=None)
    assert exc.problem_mark is None


# LLM-generated content at query #15
#--------------------------

```python
def test_yaml_is_none():
    yaml = None
    content = "example: value"
    validator = Field()
    with pytest.raises(AssertionError):
        validate_yaml(content, validator)


# LLM-generated content at query #16
#--------------------------

```python
def test_yaml_not_installed():
    import sys
    import types
    yaml_module = types.ModuleType('yaml')
    sys.modules['yaml'] = yaml_module
    yaml_module.__dict__['SafeLoader'] = None
    yaml_module.__dict__['resolver'] = None
    yaml_module.__dict__['scanner'] = None
    yaml_module.__dict__['parser'] = None
    from typesystem.tokenize import tokenize_yaml
    try:
        tokenize_yaml("")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        del sys.modules['yaml']


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        validate_yaml(content, validator)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #19
#--------------------------

```python
def test_yaml_is_none():
    import typesystem.tokenize.tokenize_yaml as module
    module.yaml = None
    with pytest.raises(AssertionError):
        module.validate_yaml("", Field())


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: {b: [1, 2, 3]}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2, 3]}}
    assert token.lookup_key(["a"]).value == "a"
    inner_dict = token.lookup(["a"])
    assert isinstance(inner_dict, DictToken)
    assert inner_dict.value == {"b": [1, 2, 3]}
    assert inner_dict.lookup_key(["b"]).value == "b"
    inner_list = inner_dict.lookup(["b"])
    assert isinstance(inner_list, ListToken)
    assert inner_list.value == [1, 2, 3]
    assert inner_list.lookup([0]).value == 1
    assert inner_list.lookup([1]).value == 2
    assert inner_list.lookup([2]).value == 3

def test_tokenize_yaml_multiline():
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=7)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(": invalid")
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #21
#--------------------------

```python
def test_yaml_parse_error_without_problem():
    exc = yaml.parser.ParserError(None)
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_with_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_with_scalar_value():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_with_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_with_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_with_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_with_bytes_content():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_with_multiline_content():
    content = """key:
  - item1
  - item2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ["item1", "item2"]}
    assert token.lookup(["key", 0]).string == "item1"
    assert token.lookup(["key", 1]).string == "item2"


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
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

def test_tokenize_yaml_boolean():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b", "c"]).value == 3
    assert token.lookup_key(["b", "c"]).value == "c"

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: c")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=7)

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    content = "key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == "value"

def test_validate_yaml_with_invalid_content():
    content = "invalid yaml content"
    validator = Field()
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert isinstance(e, ParseError)

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Field()
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert isinstance(e, ParseError)

def test_validate_yaml_with_null_validator():
    content = "key: value"
    validator = Field(allow_null=True)
    result = validate_yaml(content, validator)
    assert result == "value"

def test_validate_yaml_with_schema_validator():
    content = "key1: value1\nkey2: value2"
    validator = Schema(fields={"key1": Field(), "key2": Field()})
    result = validate_yaml(content, validator)
    assert result == {"key1": "value1", "key2": "value2"}

def test_validate_yaml_with_required_field_missing():
    content = "key1: value1"
    validator = Schema(fields={"key1": Field(), "key2": Field()})
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert isinstance(e, ValidationError)

def test_validate_yaml_with_invalid_key_type():
    content = "123: value"
    validator = Schema(fields={"key": Field()})
    try:
        validate_yaml(content, validator)
    except Exception as e:
        assert isinstance(e, ValidationError)

def test_validate_yaml_with_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": Field()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=6, char_index=5)

def test_tokenize_yaml_int():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_yaml_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_bool():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.lookup(["a", "b"]).value == 1
    assert token.lookup(["a", "c"]).value == 2

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup_key(["a"]).value == "a"
    assert isinstance(token.lookup(["a"]), ListToken)
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup_key(["b"]).value == "b"
    assert isinstance(token.lookup(["b"]), DictToken)
    assert token.lookup_key(["b", "c"]).value == "c"
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    content = "a:\n  b: c\n  d: e"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c", "d": "e"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=11)
    assert token.lookup_key(["a"]).value == "a"
    assert isinstance(token.lookup(["a"]), DictToken)
    assert token.lookup_key(["a", "b"]).value == "b"
    assert token.lookup(["a", "b"]).value == "c"
    assert token.lookup_key(["a", "d"]).value == "d"
    assert token.lookup(["a", "d"]).value == "e"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert "end of the stream" in exc_info.value.text


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_with_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: :")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_with_scalar_value():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_with_list_value():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_yaml_with_dict_value():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_with_nested_structure():
    token = tokenize_yaml("{a: {b: [1, 2]}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a"]).value == {"b": [1, 2]}
    assert token.lookup(["a", "b"]).value == [1, 2]
    assert token.lookup(["a", "b", 0]).value == 1

def test_tokenize_yaml_with_bytes_content():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_with_multiline_content():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=11)

def test_tokenize_yaml_with_special_types():
    token = tokenize_yaml("int: 42\nfloat: 3.14\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None}
    assert token.lookup(["int"]).value == 42
    assert token.lookup(["float"]).value == 3.14
    assert token.lookup(["bool"]).value is True
    assert token.lookup(["null"]).value is None


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_False():
    exc = yaml.scanner.ScannerError(problem=None, problem_mark=None)
    assert exc.problem is None


# LLM-generated content at query #5
#--------------------------

```python
def test_yaml_parse_error_without_problem():
    with pytest.raises(AssertionError):
        try:
            yaml.scanner.ScannerError(problem=None, problem_mark=None)
        except Exception as exc:
            raise exc


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_yaml_valid_content():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_content():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_bytes_content():
    content = b"key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml_empty_string_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup_key(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=4, char_index=10)

def test_tokenize_yaml_invalid_yaml_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test problem", "test context", "test note", 0, 0, 0)
    exc.problem = None
    assert exc.problem is None


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_yaml_assertion_when_pyyaml_not_installed():
    import sys
    sys.modules['yaml'] = None
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        tokenize_yaml_module.tokenize_yaml("test")


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("problem", "problem_mark")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None
    assert exc.problem_mark is None


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    global yaml
    yaml = None
    with pytest.raises(AssertionError):
        validate_yaml(content="", validator=Field())


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    import sys
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module

    # Simulate the absence of pyyaml by removing it from sys.modules
    original_modules = sys.modules.copy()
    if 'yaml' in sys.modules:
        del sys.modules['yaml']

    try:
        # Attempt to call tokenize_yaml which should raise an AssertionError
        tokenize_yaml_module.tokenize_yaml("test: value")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    finally:
        # Restore original modules to avoid side effects
        sys.modules.update(original_modules)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        validate_yaml(content, validator)


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml_valid_content():
    content = "key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == "value"

def test_validate_yaml_invalid_content():
    content = "invalid yaml content"
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
        assert "invalid yaml content" in e.text

def test_validate_yaml_empty_content():
    content = ""
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

def test_validate_yaml_null_value():
    content = "key: null"
    validator = Field(allow_null=True)
    result = validate_yaml(content, validator)
    assert result is None

def test_validate_yaml_nested_structure():
    content = "parent:\n  child: value"
    validator = Schema(fields={"parent": Schema(fields={"child": Field()})})
    result = validate_yaml(content, validator)
    assert result == {"parent": {"child": "value"}}


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3
    assert token.lookup_key(["b", "c"]).value == "c"

def test_tokenize_yaml_multiline():
    content = """
    a: 1
    b: 2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=13)
    assert token.lookup(["a"]).start == Position(line_no=2, column_no=5, char_index=5)
    assert token.lookup(["b"]).start == Position(line_no=3, column_no=5, char_index=11)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(": invalid")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.string == "[1, 2, 3]"
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.string == "{a: 1, b: 2}"
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2, 3")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.string == "hello"
    assert token.value == "hello"


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup([0]).value == [1, 2]
    assert token.lookup([1, 0]).value == 3

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [")
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_invalid_content():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        assert error.messages[0].code == "type"

def test_validate_yaml_with_missing_required_field():
    content = "key: value"
    validator = Schema(fields={"key": String(), "required_key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        assert error.messages[0].code == "required"

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "no_content"

def test_validate_yaml_with_invalid_yaml():
    content = "key: value: invalid"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ParseError as error:
        assert error.code == "parse_error"

def test_validate_yaml_with_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #21
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    global yaml
    yaml = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."
    else:
        assert False, "Expected AssertionError"


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_with_scalar_value():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_with_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert len(token._value) == 3
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_with_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_with_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_with_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{a: [1, 2,}")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_with_bytes_content():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_with_multiline_content():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.lookup(["a", "b"]).value == 1
    assert token.lookup(["a", "c"]).value == 2


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "test: value"
    validator = Field()
    assert validate_yaml(content, validator) is None


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_False():
    exc = yaml.scanner.ScannerError(problem=None, problem_mark=None)
    assert exc.problem is None


# LLM-generated content at query #25
#--------------------------

```python
def test_yaml_parse_error_without_problem():
    from typesystem.tokenize.tokenize_yaml import tokenize_yaml
    from typesystem.exceptions import ParseError

    # Mock a YAML parse error without a problem attribute
    class MockYamlError(Exception):
        def __init__(self):
            self.problem = None
            self.problem_mark = type('obj', (object,), {'index': 0})()

    with patch('typesystem.tokenize.tokenize_yaml.yaml') as mock_yaml:
        mock_yaml.load.side_effect = MockYamlError()
        try:
            tokenize_yaml("invalid yaml")
            assert False, "Expected ParseError to be raised"
        except ParseError:
            pass


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int():
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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "{a: 1, b: 2}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: c")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c"}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=7)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a", "b"]).value == "c"

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [b")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position == Position(line_no=1, column_no=5, char_index=4)


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_yaml_valid_content():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "expected '<document start>', but found '<scalar>'."

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
        assert e.messages()[0].text == "Must be a string."

def test_validate_yaml_nested_validation_error():
    content = "parent:\n  child: 123"
    validator = Schema(fields={"parent": Schema(fields={"child": String()})})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
        assert e.messages()[0].text == "Must be a string."
        assert e.messages()[0].index == ["parent", "child"]

def test_validate_yaml_missing_required_field():
    content = "key: value"
    validator = Schema(fields={"key": String(), "required_key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].text == "This field is required."
        assert e.messages()[0].index == ["required_key"]


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    yaml = None
    content = "test"
    try:
        tokenize_yaml(content)
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None


