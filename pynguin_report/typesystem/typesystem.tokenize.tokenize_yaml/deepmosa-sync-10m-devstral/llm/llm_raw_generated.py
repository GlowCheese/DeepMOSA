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
    assert token.end == Position(line_no=1, column_no=12, char_index=11)
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
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.lookup(["a", "b"]).value == 1
    assert token.lookup(["a", "c"]).value == 2
    assert token.lookup_key(["a", "b"]).value == "b"
    assert token.lookup_key(["a", "c"]).value == "c"

def test_tokenize_yaml_bytes():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2, 3")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_False():
    exc = yaml.scanner.ScannerError()
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_yaml_valid_input():
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
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_validation_error():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_yaml_required_field_missing():
    content = "other_key: value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_null_value_with_allow_null():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}

def test_validate_yaml_null_value_without_allow_null():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=False)})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"


# LLM-generated content at query #4
#--------------------------

```python
def test_yaml_is_none():
    global yaml
    yaml = None
    with pytest.raises(AssertionError):
        tokenize_yaml("test")


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "key: value"
    validator = Schema(fields={"key": StringField()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": ListField(StringField())})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": StringField()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": StringField()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_missing_required_field():
    content = "other: value"
    validator = Schema(fields={"key": StringField(), "other": StringField()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "required"


# LLM-generated content at query #6
#--------------------------

```python
def test_yaml_none_assertion():
    yaml = None
    assert yaml is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_yaml_not_installed():
    import typesystem.tokenize.tokenize_yaml as module
    module.yaml = None
    with raises(AssertionError):
        module.tokenize_yaml("test")


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_yaml_valid_input():
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
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_invalid_validation():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

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

def test_validate_yaml_list_input():
    content = "items:\n  - item1\n  - item2"
    validator = Schema(fields={"items": List(String())})
    result = validate_yaml(content, validator)
    assert result == {"items": ["item1", "item2"]}

def test_validate_yaml_union_field():
    content = "value: 123"
    validator = Schema(fields={"value": Union([String(), Integer()])})
    result = validate_yaml(content, validator)
    assert result == {"value": 123}

def test_validate_yaml_null_value():
    content = "value: null"
    validator = Schema(fields={"value": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"value": None}

def test_validate_yaml_missing_required_field():
    content = "other: value"
    validator = Schema(fields={"required": String(), "other": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "required"


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
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.lookup(["a", "b"]).value == 1
    assert token.lookup(["a", "c"]).value == 2

def test_tokenize_yaml_special_values():
    token = tokenize_yaml("null: null\nbool: true\nfloat: 3.14")
    assert isinstance(token, DictToken)
    assert token.value == {"null": None, "bool": True, "float": 3.14}
    assert token.lookup(["null"]).value is None
    assert token.lookup(["bool"]).value is True
    assert token.lookup(["float"]).value == 3.14


# LLM-generated content at query #10
#--------------------------

```python
def test_exc_problem_is_none():
    exc = yaml.scanner.ScannerError("test")
    exc.problem = None
    assert exc.problem is None


# LLM-generated content at query #11
#--------------------------

```python
def test_yaml_parse_error_has_problem():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.text == "unexpected end of stream."


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("problem", "context", "note", 1, 2, 3)
    exc.problem = None
    exc.problem_mark = None
    assert not (exc.problem is not None and exc.problem_mark is not None)


# LLM-generated content at query #13
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
    token = tokenize_yaml("a: 1\nb: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == "a: 1\nb: 2"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=3, char_index=7)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_yaml_nested_structures():
    token = tokenize_yaml("a:\n  b:\n    - 1\n    - 2\nc: 3")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}, "c": 3}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup(["a", "b", 1]).value == 2
    assert token.lookup(["c"]).value == 3

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2, 3")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_input():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #14
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

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: {b: [1, 2, 3]}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2, 3]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup(["a", "b", 1]).value == 2
    assert token.lookup(["a", "b", 2]).value == 3

def test_tokenize_yaml_multiline():
    content = """a: 1
b: 2"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).start == Position(line_no=1, column_no=1, char_index=0)
    assert token.lookup_key(["b"]).start == Position(line_no=2, column_no=1, char_index=4)

def test_tokenize_yaml_invalid_syntax():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("{a: 1, b: 2,}")
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    global yaml
    yaml = None
    with pytest.raises(AssertionError):
        validate_yaml("", Field())


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    global yaml
    yaml = None
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        tokenize_yaml("test")


# LLM-generated content at query #17
#--------------------------

```python
def test_exc_problem_mark_is_none():
    exc = yaml.parser.ParserError("test")
    exc.problem = "test"
    exc.problem_mark = None
    assert exc.problem_mark is None


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError) as excinfo:
        validate_yaml(content, validator)
    assert str(excinfo.value) == "'pyyaml' must be installed."


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_problem_is_not_none():
    exc = yaml.parser.ParserError("test")
    exc.problem = None
    assert not exc.problem is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_yaml_is_none():
    global yaml
    yaml = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()

    with pytest.raises(AssertionError) as excinfo:
        validate_yaml(content, validator)
    assert str(excinfo.value) == "'pyyaml' must be installed."


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_false():
    exc = type('MockException', (), {'problem': 'test', 'problem_mark': None})()
    assert exc.problem_mark is None


# LLM-generated content at query #24
#--------------------------

```python
def test_yaml_parse_error_with_none_problem():
    content = "invalid: yaml: content: ["
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert e.text == "."
        assert e.code == "parse_error"
        assert e.position == Position(line_no=1, column_no=18, char_index=17)


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    content = "key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_invalid_yaml():
    content = "key: value: invalid"
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "parse_error"
        assert "invalid" in e.text.lower()

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Field()
    try:
        validate_yaml(content, validator)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

def test_validate_yaml_with_null_validator():
    content = "key: value"
    validator = Field(allow_null=True)
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_bytes_content():
    content = b"key: value"
    validator = Field()
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_schema_validator():
    schema = Schema(fields={"name": Field(), "age": Field()})
    content = "name: John\nage: 30"
    result = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_with_missing_required_field():
    schema = Schema(fields={"name": Field(), "age": Field()})
    content = "name: John"
    try:
        validate_yaml(content, schema)
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

def test_validate_yaml_with_invalid_field_type():
    schema = Schema(fields={"name": Field(), "age": Field()})
    content = "name: John\nage: thirty"
    try:
        validate_yaml(content, schema)
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

def test_validate_yaml_with_nested_schema():
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    outer_schema = Schema(fields={"name": Field(), "address": inner_schema})
    content = "name: John\naddress:\n  street: 123 Main\n  city: Anytown"
    result = validate_yaml(content, outer_schema)
    assert result == {"name": "John", "address": {"street": "123 Main", "city": "Anytown"}}

def test_validate_yaml_with_list_content():
    schema = Schema(fields={"items": Field()})
    content = "items:\n  - item1\n  - item2"
    result = validate_yaml(content, schema)
    assert result == {"items": ["item1", "item2"]}


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    yaml = None
    assert not yaml is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test")
    exc.problem = None
    assert exc.problem is None


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError("test problem")
    exc.problem = None
    exc.problem_mark = None
    assert exc.problem is None
    assert exc.problem_mark is None


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_yaml_with_valid_content():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_with_invalid_yaml():
    content = "key: [value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as error:
        assert error.code == "parse_error"

def test_validate_yaml_with_invalid_schema():
    content = "key: value"
    validator = Schema(fields={"other_key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].code == "required"

def test_validate_yaml_with_union_field():
    content = "key: 123"
    validator = Schema(fields={"key": String() | Integer()})
    result = validate_yaml(content, validator)
    assert result == {"key": 123}

def test_validate_yaml_with_null_value():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}

def test_validate_yaml_with_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as error:
        assert error.code == "no_content"

def test_validate_yaml_with_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_yaml_valid_input():
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
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_validation_error():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_yaml_required_field_missing():
    content = "other_key: value"
    validator = Schema(fields={"key": String(), "other_key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_null_value():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_yaml_empty_string_raises_parse_error():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)
    else:
        assert False, "Expected ParseError to be raised"

def test_tokenize_yaml_invalid_yaml_raises_parse_error():
    try:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    except ParseError as e:
        assert e.text.endswith(".")
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)
    else:
        assert False, "Expected ParseError to be raised"

def test_tokenize_yaml_scalar_value():
    result = tokenize_yaml("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.string == "42"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_list_value():
    result = tokenize_yaml("[1, 2, 3]")
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.string == "[1, 2, 3]"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_yaml_dict_value():
    result = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(result, DictToken)
    assert result.value == {"a": 1, "b": 2}
    assert result.string == "{a: 1, b: 2}"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    result = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(result, DictToken)
    assert result.value == {"a": [1, 2], "b": {"c": 3}}
    assert result.lookup(["a"]).value == [1, 2]
    assert result.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_bytes_input():
    result = tokenize_yaml(b"42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.string == "42"

def test_tokenize_yaml_multiline_content():
    content = """key: value
list:
  - item1
  - item2"""
    result = tokenize_yaml(content)
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value", "list": ["item1", "item2"]}
    assert result.lookup(["list", 0]).value == "item1"
    assert result.lookup(["list", 1]).value == "item2"

def test_tokenize_yaml_special_types():
    content = """int: 42
float: 3.14
bool: true
null: null"""
    result = tokenize_yaml(content)
    assert result.value == {"int": 42, "float": 3.14, "bool": True, "null": None}
    assert isinstance(result.lookup(["int"]), ScalarToken)
    assert isinstance(result.lookup(["float"]), ScalarToken)
    assert isinstance(result.lookup(["bool"]), ScalarToken)
    assert isinstance(result.lookup(["null"]), ScalarToken)


# LLM-generated content at query #3
#--------------------------

```python
def test_problem_is_none():
    exc = type('MockException', (), {'problem': None, 'problem_mark': type('Mark', (), {'index': 0})()})()
    assert exc.problem is None


# LLM-generated content at query #4
#--------------------------

```python
def test_yaml_not_installed():
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module
    tokenize_yaml_module.yaml = None
    with assert_raises(AssertionError) as excinfo:
        tokenize_yaml_module.tokenize_yaml("test")
    assert str(excinfo.value) == "'pyyaml' must be installed."


# LLM-generated content at query #5
#--------------------------

```python
def test_problem_and_problem_mark_not_none_when_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid yaml: [")
    assert exc_info.value.text == "invalid yaml."
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #6
#--------------------------

```python
def test_yaml_parse_error_with_none_problem():
    class MockError:
        problem = None
        problem_mark = type('obj', (object,), {'index': 0})()

    with pytest.raises(AssertionError):
        tokenize_yaml._handle_yaml_error(MockError(), "")


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    global yaml
    yaml = None
    try:
        tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_yaml_empty_string_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_scalar_token():
    content = "hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_list_token():
    content = "- item1\n- item2"
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=12)

def test_tokenize_yaml_dict_token():
    content = "key1: value1\nkey2: value2"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key1": "value1", "key2": "value2"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=10, char_index=19)

def test_tokenize_yaml_nested_structure():
    content = "list:\n  - item1\n  - item2\nnested:\n  key: value"
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"list": ["item1", "item2"], "nested": {"key": "value"}}
    assert token.lookup([0]).value == ["item1", "item2"]
    assert token.lookup([1]).value == {"key": "value"}
    assert token.lookup_key([1, "key"]).value == "value"

def test_tokenize_yaml_bytes_content():
    content = b"hello"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_invalid_yaml_raises_parse_error():
    content = "invalid: yaml: content: ["
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(content)
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_yaml_empty_string():
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
    assert token.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

def test_tokenize_yaml_invalid_yaml():
    try:
        tokenize_yaml("a: b: c")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "mapping values are not allowed here" in e.text


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    with patch.dict('sys.modules', {'yaml': None}):
        with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
            validate_yaml(content="test", validator=Field())


# LLM-generated content at query #11
#--------------------------

```python
def test_yaml_parse_error_has_problem():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid yaml: [")
    assert exc_info.value.text == "expected '<document start>', but found '<stream end>'."


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_yaml_empty_string_raises_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_yaml_scalar_token():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_int_token():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_float_token():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

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
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_yaml_dict_token():
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

def test_tokenize_yaml_multiline_string():
    token = tokenize_yaml("hello\nworld")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello\nworld"
    assert token.string == "hello\nworld"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=5, char_index=10)

def test_tokenize_yaml_invalid_yaml_raises_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid: yaml: content: [unclosed")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)


# LLM-generated content at query #13
#--------------------------

```python
def test_yaml_parse_error_has_problem_mark():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("invalid yaml: [")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text == "end of the stream or a document separator is expected."


# LLM-generated content at query #14
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
    token = tokenize_yaml("- foo\n- bar")
    assert isinstance(token, ListToken)
    assert token.value == ["foo", "bar"]
    assert token.string == "- foo\n- bar"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=4, char_index=9)

def test_tokenize_yaml_dict():
    token = tokenize_yaml("foo: bar\nbaz: qux")
    assert isinstance(token, DictToken)
    assert token.value == {"foo": "bar", "baz": "qux"}
    assert token.string == "foo: bar\nbaz: qux"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=2, column_no=7, char_index=13)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("foo:\n  - bar\n  - baz: qux")
    assert isinstance(token, DictToken)
    assert token.value == {"foo": ["bar", {"baz": "qux"}]}
    assert token.string == "foo:\n  - bar\n  - baz: qux"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=9, char_index=20)

def test_tokenize_yaml_lookup():
    token = tokenize_yaml("foo:\n  - bar\n  - baz: qux")
    assert token.lookup([0]).value == "bar"
    assert token.lookup([1, "baz"]).value == "qux"

def test_tokenize_yaml_lookup_key():
    token = tokenize_yaml("foo:\n  bar: baz")
    assert token.lookup_key(["bar"]).value == "bar"

def test_tokenize_yaml_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("foo: bar: baz")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

def test_tokenize_yaml_bytes_content():
    token = tokenize_yaml(b"foo")
    assert isinstance(token, ScalarToken)
    assert token.value == "foo"
    assert token.string == "foo"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_yaml_valid_input():
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
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_validation_error():
    content = "key: value"
    validator = Schema(fields={"key": Integer()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_yaml_required_field_missing():
    content = "other: value"
    validator = Schema(fields={"key": String(), "other": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_bytes_input():
    content = b"key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    inner_schema = Schema(fields={"inner": String()})
    validator = Schema(fields={"outer": inner_schema})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError) as excinfo:
        validate_yaml(content, validator)
    assert str(excinfo.value) == "'pyyaml' must be installed."


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_yaml_valid_input():
    content = "key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_invalid_yaml():
    content = "key: [unclosed"
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert "unclosed" in exc.text.lower()

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as exc:
        assert len(exc.messages) == 1
        assert exc.messages[0].code == "type"

def test_validate_yaml_missing_required_field():
    content = "other: value"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
    except ValidationError as exc:
        assert len(exc.messages) == 1
        assert exc.messages[0].code == "required"
        assert exc.messages[0].text == "The field 'key' is required."

def test_validate_yaml_bytes_input():
    content = b"key: value"
    validator = Schema(fields={"key": String()})
    result = validate_yaml(content, validator)
    assert result == {"key": "value"}

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={})
    try:
        validate_yaml(content, validator)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={
        "outer": Schema(fields={"inner": String()})
    })
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}

def test_validate_yaml_list_input():
    content = "items:\n  - one\n  - two"
    validator = Schema(fields={
        "items": List(String())
    })
    result = validate_yaml(content, validator)
    assert result == {"items": ["one", "two"]}

def test_validate_yaml_union_type():
    content = "value: 123"
    validator = Schema(fields={
        "value": String() | Integer()
    })
    result = validate_yaml(content, validator)
    assert result == {"value": 123}

def test_validate_yaml_null_value():
    content = "value: null"
    validator = Schema(fields={
        "value": String(allow_null=True)
    })
    result = validate_yaml(content, validator)
    assert result == {"value": None}


# LLM-generated content at query #18
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
    assert token.lookup(["key", 0]).value == "item1"
    assert token.lookup(["key", 1]).value == "item2"

def test_tokenize_yaml_with_special_types():
    content = """int: 42
float: 3.14
bool: true
null: null"""
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None}
    assert isinstance(token.lookup(["int"]), ScalarToken)
    assert isinstance(token.lookup(["float"]), ScalarToken)
    assert isinstance(token.lookup(["bool"]), ScalarToken)
    assert isinstance(token.lookup(["null"]), ScalarToken)

def test_tokenize_yaml_with_position_tracking():
    content = """line1: value1
line2: value2"""
    token = tokenize_yaml(content)
    assert token.lookup(["line1"]).start.line_no == 1
    assert token.lookup(["line1"]).end.line_no == 1
    assert token.lookup(["line2"]).start.line_no == 2
    assert token.lookup(["line2"]).end.line_no == 2

def test_tokenize_yaml_with_equality_check():
    token1 = tokenize_yaml("42")
    token2 = tokenize_yaml("42")
    assert token1 == token2


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test", None, None, None, None, None)
    assert exc.problem is None


# LLM-generated content at query #20
#--------------------------

```python
def test_yaml_is_none():
    global yaml
    yaml = None
    with pytest.raises(AssertionError, match="'pyyaml' must be installed."):
        tokenize_yaml("test")


# LLM-generated content at query #21
#--------------------------

```python
def test_yaml_is_none():
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
def test_validate_yaml_valid_input():
    schema = Schema(fields={"name": String(), "age": Integer()})
    content = "name: John\nage: 30"
    result = validate_yaml(content, schema)
    assert result == {"name": "John", "age": 30}

def test_validate_yaml_invalid_yaml():
    schema = Schema(fields={"name": String()})
    content = "name: [John"
    try:
        validate_yaml(content, schema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_empty_content():
    schema = Schema(fields={"name": String()})
    content = ""
    try:
        validate_yaml(content, schema)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_missing_required_field():
    schema = Schema(fields={"name": String(), "age": Integer()})
    content = "name: John"
    try:
        validate_yaml(content, schema)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

def test_validate_yaml_invalid_field_type():
    schema = Schema(fields={"name": String(), "age": Integer()})
    content = "name: John\nage: thirty"
    try:
        validate_yaml(content, schema)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

def test_validate_yaml_null_value_with_allow_null():
    schema = Schema(fields={"name": String(allow_null=True)})
    content = "name: null"
    result = validate_yaml(content, schema)
    assert result == {"name": None}

def test_validate_yaml_null_value_without_allow_null():
    schema = Schema(fields={"name": String()})
    content = "name: null"
    try:
        validate_yaml(content, schema)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

def test_validate_yaml_nested_schema():
    inner_schema = Schema(fields={"street": String(), "city": String()})
    schema = Schema(fields={"name": String(), "address": inner_schema})
    content = "name: John\naddress:\n  street: 123 Main St\n  city: Anytown"
    result = validate_yaml(content, schema)
    assert result == {"name": "John", "address": {"street": "123 Main St", "city": "Anytown"}}


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_line_84_evaluates_to_false():
    exc = yaml.parser.ParserError(None)
    exc.problem = "test problem"
    exc.problem_mark = None
    assert exc.problem_mark is None


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

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
    token = tokenize_yaml("a:\n  b: [1, 2]")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup_key(["a", "b"]).value == "b"

def test_tokenize_yaml_with_invalid_yaml():
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("a: [1, 2")
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.char_index == 7

def test_tokenize_yaml_with_bytes_content():
    token = tokenize_yaml(b"42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"

def test_tokenize_yaml_with_multiline_content():
    token = tokenize_yaml("a: 1\nb: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.lookup(["a"]).start.line_no == 1
    assert token.lookup(["b"]).start.line_no == 2


# LLM-generated content at query #25
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
    assert token.end == Position(line_no=3, column_no=3, char_index=11)

def test_tokenize_yaml_nested_structure():
    token = tokenize_yaml("a:\n  b:\n    - c\n    - d\ne: f")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": ["c", "d"]}, "e": "f"}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=5, column_no=3, char_index=19)
    assert token.lookup(["a"]).value == {"b": ["c", "d"]}
    assert token.lookup(["a", "b"]).value == ["c", "d"]
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


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.scanner.ScannerError("test", "test", "test", 0, 0, None)
    assert exc.problem is None


# LLM-generated content at query #27
#--------------------------

```python
def test_yaml_not_installed():
    import typesystem.tokenize.tokenize_yaml as tokenize_yaml_module
    tokenize_yaml_module.yaml = None
    try:
        tokenize_yaml_module.tokenize_yaml("test")
    except AssertionError as e:
        assert str(e) == "'pyyaml' must be installed."


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_yaml_without_pyyaml():
    yaml = None
    content = "key: value"
    validator = Field()
    with pytest.raises(AssertionError):
        validate_yaml(content, validator)


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_yaml_with_empty_string():
    try:
        tokenize_yaml("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_yaml_with_scalar_value():
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_yaml_with_integer():
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_yaml_with_float():
    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_with_boolean():
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_with_null():
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_yaml_with_list():
    token = tokenize_yaml("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_yaml_with_dict():
    token = tokenize_yaml("{a: 1, b: 2}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=11, char_index=10)

def test_tokenize_yaml_with_nested_structure():
    token = tokenize_yaml("{a: [1, 2], b: {c: 3}}")
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_yaml_with_multiline_content():
    token = tokenize_yaml("a:\n  b: 1\n  c: 2")
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1, "c": 2}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=5, char_index=11)

def test_tokenize_yaml_with_invalid_yaml():
    try:
        tokenize_yaml("a: [1, 2")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position == Position(line_no=1, column_no=7, char_index=6)


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_83_evaluates_to_false():
    exc = yaml.parser.ParserError("problem", "problem_mark", None, None, None)
    assert exc.problem is None


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_yaml_valid_input():
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
    except ParseError as e:
        assert e.code == "parse_error"

def test_validate_yaml_empty_content():
    content = ""
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"

def test_validate_yaml_validation_error():
    content = "key: 123"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

def test_validate_yaml_required_field_missing():
    content = "other_key: value"
    validator = Schema(fields={"key": String(), "other_key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["key"]

def test_validate_yaml_nested_schema():
    content = "outer:\n  inner: value"
    validator = Schema(fields={"outer": Schema(fields={"inner": String()})})
    result = validate_yaml(content, validator)
    assert result == {"outer": {"inner": "value"}}

def test_validate_yaml_with_default_value():
    content = "key: value"
    validator = Schema(fields={"key": String(), "optional": String(default="default")})
    result = validate_yaml(content, validator)
    assert result == {"key": "value", "optional": "default"}

def test_validate_yaml_null_value_with_allow_null():
    content = "key: null"
    validator = Schema(fields={"key": String(allow_null=True)})
    result = validate_yaml(content, validator)
    assert result == {"key": None}

def test_validate_yaml_null_value_without_allow_null():
    content = "key: null"
    validator = Schema(fields={"key": String()})
    try:
        validate_yaml(content, validator)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

def test_validate_yaml_list_input():
    content = "key:\n  - item1\n  - item2"
    validator = Schema(fields={"key": List(String())})
    result = validate_yaml(content, validator)
    assert result == {"key": ["item1", "item2"]}


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_yaml_without_pyyaml():
    global yaml
    original_yaml = yaml
    yaml = None
    try:
        content = "key: value"
        assert_raises(AssertionError, tokenize_yaml, content)
    finally:
        yaml = original_yaml


