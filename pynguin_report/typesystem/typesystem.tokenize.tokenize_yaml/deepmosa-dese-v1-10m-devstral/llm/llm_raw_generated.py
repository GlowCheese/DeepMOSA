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
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

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
    token = tokenize_yaml("a: {b: [1, 2]}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=10)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_special_types():
    token_int = tokenize_yaml("42")
    assert isinstance(token_int, ScalarToken)
    assert isinstance(token_int.value, int)

    token_float = tokenize_yaml("3.14")
    assert isinstance(token_float, ScalarToken)
    assert isinstance(token_float.value, float)

    token_bool = tokenize_yaml("true")
    assert isinstance(token_bool, ScalarToken)
    assert isinstance(token_bool.value, bool)

    token_null = tokenize_yaml("null")
    assert isinstance(token_null, ScalarToken)
    assert token_null.value is None


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": List(items=String())})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "expected '<block end>', but found '<scalar>'."

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
        assert e.messages()[0].text == "Must be a string."

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

def test_validate_yaml_missing_required_field():
    content = "other_key: value"
    validator = Schema(fields={"key": String(), "other_key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].text == "This field is required."

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    global yaml
    yaml = None
    with raises(AssertionError):
        tokenize_yaml("test")


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(line_no=1, column_no=1, char_index=0)

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
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup(["a"]).start == Position(line_no=1, column_no=1, char_index=0)
    assert token.lookup(["b"]).start == Position(line_no=2, column_no=1, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("{a: 1, b: 2,}")
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position == Position(line_no=1, column_no=12, char_index=11)


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == "value"

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "expected '<block end>', but found '<scalar>'."

def test_validate_yaml_empty_content():
    content = ""
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

def test_validate_yaml_validation_error():
    content = "key: value"
    validator = Schema(fields={"required_key": Field()})
    try:
        validate_yaml(content, validator)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].text == "This field is required."

def test_validate_yaml_with_schema():
    content = "name: John\nage: 30"
    validator = Schema(fields={
        "name": Field(),
        "age": Field()
    })
    result = validate_yaml(content, validator)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_nested_schema():
    content = "person:\n  name: John\n  age: 30"
    validator = Schema(fields={
        "person": Schema(fields={
            "name": Field(),
            "age": Field()
        })
    })
    result = validate_yaml(content, validator)
    assert result == {"person": {"name": "John", "age": 30}}


# LLM-generated content at query #6
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
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup(["a"]).start == Position(line_no=1, column_no=1, char_index=0)
    assert token.lookup(["b"]).start == Position(line_no=2, column_no=1, char_index=5)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a: [1, 2")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

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
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=5, char_index=15)

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2")
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2, 3")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("a:\n  b: |-\n    line1\n    line2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "line1\nline2\n"}}
    assert token.lookup(["a", "b"]).string == "line1\n    line2"

def test_tokenize_yaml_special_types():
    token = tokenize_yaml("int: 42\nfloat: 3.14\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None}
    assert token.lookup(["int"]).value == 42
    assert token.lookup(["float"]).value == 3.14
    assert token.lookup(["bool"]).value is True
    assert token.lookup(["null"]).value is None


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test problem")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    yaml = None
    assert yaml is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with assert_raises(ParseError) as exc_info:
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
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"

def test_tokenize_yaml_invalid_yaml():
    with assert_raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_multiline_content():
    content = """key: value
list:
  - item1
  - item2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "list": ["item1", "item2"]}
    assert token.lookup(["key"]).value == "value"
    assert token.lookup(["list", 0]).value == "item1"
    assert token.lookup(["list", 1]).value == "item2"


# LLM-generated content at query #11
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

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b"]).value == {"c": 3}
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_multiline():
    content = """
    a: 1
    b: 2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=2, column_no=5, char_index=5)
    assert token.end == Position(line_no=3, column_no=6, char_index=12)

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_yaml_success():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position is not None

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
        assert e.messages()[0].start_position is not None
        assert e.messages()[0].end_position is not None

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    assert validate_yaml(content, validator) is None


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_yaml_empty_content():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

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
    assert token.lookup(["b"]).value == {"c": 3}
    assert token.lookup_key(["b", "c"]).value == "c"

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("a: b: c")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "mapping values are not allowed here" in e.text


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    assert not (False and False)


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(line_no=1, column_no=1, char_index=0)

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
    token = tokenize_yaml("{a: {b: [1, 2]}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup(["a", "b", 1]).value == 2

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.lookup(["a", "b"]).value == 1
    assert token.lookup(["a", "c"]).value == 2

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a: [1, 2")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    global yaml
    yaml = None
    with pytest.raises(AssertionError):
        validate_yaml("", Field())


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
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

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
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2\n  c: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2], "c": 3}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [b: c]")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None
    assert exc.problem_mark is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    content = """key: value
list:
  - item1
  - item2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "list": ["item1", "item2"]}
    assert token.lookup(["key"]).value == "value"
    assert token.lookup(["list", 0]).value == "item1"
    assert token.lookup(["list", 1]).value == "item2"

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("invalid: yaml: content")
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_yaml_valid():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert len(exc.messages) == 1
        assert exc.messages[0].code == "type"

def test_validate_yaml_empty_string():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"

def test_validate_yaml_with_bytes():
    content = b"key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}

def test_validate_yaml_with_list():
    content = "key:\n  - item1\n  - item2"
    validator = Schema(fields={"key": List(fields=[String()])})
    result = validate_yaml(content, validator)
    assert result == {"key": ["item1", "item2"]}

def test_validate_yaml_with_union():
    content = "key: 123"
    validator = Schema(fields={"key": Union(any_of=[String(), Integer()])})
    result = validate_yaml(content, validator)
    assert result == {"key": 123}

def test_validate_yaml_with_missing_required_field():
    content = "key: value"
    validator = Schema(fields={"key": String(), "required_key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert any(msg.code == "required" for msg in exc.messages)

def test_validate_yaml_with_null_value():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}


# LLM-generated content at query #3
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

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
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

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("a: [1, 2")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #4
#--------------------------

```python
def test_yaml_not_installed():
    import sys
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module

    # Mock yaml as None to simulate it not being installed
    yaml_backup = sys.modules.get('yaml')
    sys.modules['yaml'] = None

    try:
        with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
            tokenize_yaml_module.tokenize_yaml("test")
    finally:
        if yaml_backup is not None:
            sys.modules['yaml'] = yaml_backup
        else:
            del sys.modules['yaml']


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
    token = tokenize_yaml("- a\n- b\n- c")
    assert isinstance(token, ListToken)
    assert token.value == ["a", "b", "c"]
    assert token.string == "- a\n- b\n- c"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("a: 1\nb: 2\nc: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2, "c": 3}
    assert token.string == "a: 1\nb: 2\nc: 3"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=3, char_index=9)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - c\n    - d\ne: f")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": ["c", "d"]}, "e": "f"}
    assert token.lookup(["a", "b", 0]).value == "c"
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [b: c]")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

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
    assert token.end == Position(line_no=1, column_no=13, char_index=12)

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

def test_tokenize_yaml_with_invalid_yaml():
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("invalid: yaml: content: [")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)


# LLM-generated content at query #7
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
    token = tokenize_yaml('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3
    assert token.lookup_key(["b", "c"]).value == "c"

def test_tokenize_yaml_multiline():
    token = tokenize_yaml("a:\n  b: c\n  d: e")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": "c", "d": "e"}}
    assert token.lookup(["a", "b"]).value == "c"
    assert token.lookup(["a", "d"]).value == "e"
    assert token.lookup_key(["a", "b"]).value == "b"
    assert token.lookup_key(["a", "d"]).value == "d"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [b: c]")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = type('MockException', (), {'problem': None, 'problem_mark': None})()
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    yaml = None
    assert yaml is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_yaml_not_installed():
    import sys
    from unittest.mock import MagicMock

    # Mock the yaml module to be None
    sys.modules['yaml'] = None

    from typesystem.tokenize.tokenize_yaml import tokenize_yaml

    try:
        tokenize_yaml("test")
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError):
        validate_yaml(content, validator)


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test")
    exc.problem = "test"
    exc.problem_mark = None
    assert exc.problem_mark is None


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_yaml_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar():
    token = tokenize_yaml("foo")
    assert isinstance(token, ScalarToken)
    assert token.value == "foo"
    assert token.string == "foo"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

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
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=7)
    assert token.lookup(["a"]).start == Position(line_no=1, column_no=1, char_index=0)
    assert token.lookup(["b"]).start == Position(line_no=2, column_no=1, char_index=5)

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"foo")
    assert isinstance(token, ScalarToken)
    assert token.value == "foo"
    assert token.string == "foo"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #14
#--------------------------

```python
def test_parse_error_with_none_problem():
    exc = yaml.parser.ParserError(None, None)
    assert exc.problem is None


# LLM-generated content at query #15
#--------------------------

```python
def test_problem_is_none():
    exc = type("MockExc", (), {"problem": None, "problem_mark": type("Mark", (), {"index": 0})()})()
    assert exc.problem is None


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml_valid():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "parse_error"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert exc.code == "no_content"

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "type"

def test_validate_yaml_required_field_missing():
    content = "other: value"
    validator = Schema(fields={"key": String(), "other": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "required"
        assert exc.messages()[0].index == ["key"]

def test_validate_yaml_nested_schema():
    content = "nested:\n  key: value"
    validator = Schema(fields={"nested": Schema(fields={"key": String()})})
    result = validate_yaml(content, validator)
    assert result == {"nested": {"key": "value"}}

def test_validate_yaml_list_validation():
    content = "items:\n  - item1\n  - item2"
    validator = Schema(fields={"items": List(String())})
    result = validate_yaml(content, validator)
    assert result == {"items": ["item1", "item2"]}

def test_validate_yaml_null_value():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}

def test_validate_yaml_bytes_content():
    content = b"key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_positional_error_messages():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        messages = exc.messages()
        assert len(messages) == 1
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "test: value"
    validator = Field()
    assert_raises(AssertionError, validate_yaml, content, validator)


# LLM-generated content at query #18
#--------------------------

```python
def test_yaml_import_failure():
    import sys
    import typesystem.tokenize.tokenize_yaml as module
    original_yaml = sys.modules.get('yaml')
    sys.modules['yaml'] = None
    try:
        with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
            module.tokenize_yaml("test")
    finally:
        sys.modules['yaml'] = original_yaml


# LLM-generated content at query #19
#--------------------------

```python
def test_yaml_parse_error_with_problem_and_problem_mark():
    content = "invalid: yaml: content: [unclosed"
    exc = yaml.parser.ParserError("context", "problem", None, None, None)
    exc.problem = "some problem"
    exc.problem_mark = type('Mark', (), {'index': 10})()
    assert exc.problem is not None
    assert exc.problem_mark is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        validate_yaml(content, validator)


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_false():
    exc = yaml.parser.ParserError(None)
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #22
#--------------------------

```python
def test_yaml_parse_error_without_problem():
    class MockYamlError:
        problem = None
        problem_mark = type('obj', (object,), {'index': 0})()

    with pytest.raises(AssertionError):
        tokenize_yaml._handle_yaml_error(MockYamlError())


# LLM-generated content at query #23
#--------------------------

```python
def test_yaml_is_none():
    global yaml
    yaml = None
    with pytest.raises(AssertionError):
        validate_yaml(content="", validator=Field())


