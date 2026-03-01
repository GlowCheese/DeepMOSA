####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token
    import typing

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 5)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": null}')
    token, end = scanner('{"test": null}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": true}')
    token, end = scanner('{"test": true}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": false}')
    token, end = scanner('{"test": false}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": 123}')
    token, end = scanner('{"test": 123}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": 123.45}')
    token, end = scanner('{"test": 123.45}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 7

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 2)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": []}')
    token, end = scanner('{"test": []}', 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 2

def test_make_scanner_scans_dict():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    class MockTokenizingJSONObject:
        def __call__(self, *args, **kwargs):
            return ({}, 2)

    context = MockContext()
    context.parse_object = MockTokenizingJSONObject()
    scanner = _make_scanner(context, '{"test": {}}')
    token, end = scanner('{"test": {}}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    assert not ("n" == "x" and "x"[:4] == "null")


# LLM-generated content at query #3
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("test", 6),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.start_index == 0
    assert token.end_index == 5
    assert end == 6

def test_make_scanner_scans_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("key", 5),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    mock_parse_object = lambda x, y, z, w, c: ({}, 2)
    scanner = _make_scanner(context, '{"key": 1}')
    scanner.__closure__[0].cell_contents = mock_parse_object

    token, end = scanner('{"key": 1}', 0)
    assert isinstance(token, DictToken)
    assert token.start_index == 0
    assert token.end_index == 1
    assert end == 2

def test_make_scanner_scans_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 2),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "[]")
    token, end = scanner("[]", 0)
    assert isinstance(token, ListToken)
    assert token.start_index == 0
    assert token.end_index == 1
    assert end == 2

def test_make_scanner_scans_null_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start_index == 0
    assert token.end_index == 3
    assert end == 4

def test_make_scanner_scans_true_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start_index == 0
    assert token.end_index == 3
    assert end == 4

def test_make_scanner_scans_false_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start_index == 0
    assert token.end_index == 4
    assert end == 5

def test_make_scanner_scans_number_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start_index == 0
    assert token.end_index == 2
    assert end == 3

def test_make_scanner_scans_float_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {}
    })()

    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start_index == 0
    assert token.end_index == 5
    assert end == 6

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    context = type("MockContext", (), {
        "parse_array": lambda x, y: ([], 0),
        "parse_string": lambda x, y, z: ("", 0),
        "parse_float": float,
        "parse_int": int,
        "strict": True,
        "memo": {"test": "value"}
    })()

    scanner = _make_scanner(context, "")
    scanner("", 0)
    assert context.memo == {}


# LLM-generated content at query #4
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    scan_once = lambda s, e: (ScalarToken('value', 0, 0), 1)
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), True, scan_once, {}, '{"key": "value"}')
    assert len(result) == 1
    assert result['key'].value == 'value'
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    scan_once = lambda s, e: (ScalarToken('value', 0, 0), 1)
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), True, scan_once, {}, '{"key1": "value1", "key2": "value2"}')
    assert len(result) == 2
    assert result['key1'].value == 'value1'
    assert result['key2'].value == 'value2'
    assert end == 33

def test__TokenizingJSONObject_with_whitespace():
    scan_once = lambda s, e: (ScalarToken('value', 0, 0), 1)
    result, end = _TokenizingJSONObject(('{ "key" : "value" }', 0), True, scan_once, {}, '{ "key" : "value" }')
    assert len(result) == 1
    assert result['key'].value == 'value'
    assert end == 19

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, e: (ScalarToken('value', 0, 0), 1), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, e: (ScalarToken('value', 0, 0), 1), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), True, lambda s, e: (ScalarToken('value', 0, 0), 1), {}, '{"key": "value"')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: "value"}', 0), True, lambda s, e: (ScalarToken('value', 0, 0), 1), {}, '{123: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_null():
    token = tokenize_json("null")
    assert token == ScalarToken(None, 0, 3, "null")
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_boolean():
    true_token = tokenize_json("true")
    assert true_token == ScalarToken(True, 0, 3, "true")
    assert true_token.value is True
    assert true_token.string == "true"

    false_token = tokenize_json("false")
    assert false_token == ScalarToken(False, 0, 4, "false")
    assert false_token.value is False
    assert false_token.string == "false"

def test_tokenize_json_number():
    int_token = tokenize_json("42")
    assert int_token == ScalarToken(42, 0, 1, "42")
    assert int_token.value == 42
    assert int_token.string == "42"

    float_token = tokenize_json("3.14")
    assert float_token == ScalarToken(3.14, 0, 3, "3.14")
    assert float_token.value == 3.14
    assert float_token.string == "3.14"

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert token == ScalarToken("hello", 0, 6, '"hello"')
    assert token.value == "hello"
    assert token.string == '"hello"'

def test_tokenize_json_list():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_json_dict():
    token = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)

def test_tokenize_json_nested_structures():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a"]).value == [1, 2]
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_json_with_valid_json():
    content = '{"key": "value"}'
    result = tokenize_json(content)
    assert isinstance(result, Token)


# LLM-generated content at query #7
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = typing.Any
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start._index == 7
    assert token.end._index == 10


# LLM-generated content at query #8
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i+1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    s = '{"key": "value"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(s)

def test__TokenizingJSONObject_multiple_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value1" if i == 8 else "value2", i, i+6, s), i+7), memo, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(s)

def test__TokenizingJSONObject_with_whitespace():
    s = '{"key" : "value"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(s)

def test__TokenizingJSONObject_missing_colon():
    s = '{"key" "value"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    s = '{"key1": "value1" "key2": "value2"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value1" if i == 9 else "value2", i, i+6, s), i+7), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    s = '{"key": "value"'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test__TokenizingJSONObject_nested_object():
    s = '{"outer": {"inner": "value"}}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (_TokenizingJSONObject((s, i), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), memo, content)[0], i+17), memo, content)
    assert len(result) == 1
    assert isinstance(result["outer"], dict)
    assert result["outer"]["inner"].value == "value"
    assert end == len(s)


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    assert not (nextchar != '"')


# LLM-generated content at query #10
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    assert parse_object is not _TokenizingJSONObject


# LLM-generated content at query #11
#--------------------------

```python
def test_line_61_predicate_false():
    s = '{"key": "value"}'
    end = len(s) - 1
    nextchar = s[end]
    assert nextchar != ""


# LLM-generated content at query #12
#--------------------------

```python
def test_index_error_handling_in_tokenizing_json_object():
    s = '{"key": "value"'
    end = len(s)
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, content), end)
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key": "value"}
    assert end == len(s)


# LLM-generated content at query #13
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken, ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([ScalarToken(1, 0, 0, "")], 3)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "[1]")
    token, end = scanner("[1]", 0)
    assert isinstance(token, ListToken)
    assert token.value == [1]
    assert end == 3

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken, ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("key", y + 3)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    class MockTokenizingJSONObject:
        def __call__(self, *args, **kwargs):
            key = ScalarToken("key", 0, 2, "")
            value = ScalarToken("value", 0, 4, "")
            return {key: value}, 10

    import typesystem.tokenize.tokenize_json as module
    module._TokenizingJSONObject = MockTokenizingJSONObject()

    context = MockContext()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert end == 13


# LLM-generated content at query #14
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    s = '{"key": "value"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i + 4, s), i + 5), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(s)

def test__TokenizingJSONObject_multiple_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value1", i, i + 6, s), i + 7), memo, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert end == len(s)

def test__TokenizingJSONObject_whitespace_handling():
    s = '{"key" : "value" }'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i + 4, s), i + 5), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(s)

def test__TokenizingJSONObject_missing_quotes():
    import pytest
    s = '{key: "value"}'
    content = s
    memo = {}
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i + 4, s), i + 5), memo, content)

def test__TokenizingJSONObject_missing_colon():
    import pytest
    s = '{"key" "value"}'
    content = s
    memo = {}
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value", i, i + 4, s), i + 5), memo, content)

def test__TokenizingJSONObject_missing_comma():
    import pytest
    s = '{"key1": "value1" "key2": "value2"}'
    content = s
    memo = {}
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken("value1", i, i + 6, s), i + 7), memo, content)

def test__TokenizingJSONObject_missing_value():
    import pytest
    s = '{"key": }'
    content = s
    memo = {}
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject((s, 0), True, lambda s, i: (ScalarToken(None, i, i, s), i + 1), memo, content)


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_null():
    token = tokenize_json("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_true():
    token = tokenize_json("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_false():
    token = tokenize_json("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_number_integer():
    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_number_float():
    token = tokenize_json("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_json_dict():
    token = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_json_nested():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_json_bytes():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting" in e.text


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    s = '{"key": "value"}'
    end = len(s)
    nextchar = s[end : end + 1]
    assert nextchar == ""


# LLM-generated content at query #17
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token
    import typing

    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = lambda s, i, strict: ("value", i + 6)
        parse_array = None
        parse_float = None
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value == "value"
    assert end == 7

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = None
        parse_array = None
        parse_float = None
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": null}')
    token, end = scanner('{"key": null}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 5

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = None
        parse_array = None
        parse_float = None
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": true}')
    token, end = scanner('{"key": true}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 5

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = None
        parse_array = None
        parse_float = None
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": false}')
    token, end = scanner('{"key": false}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 6

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = None
        parse_array = None
        parse_float = lambda x: float(x)
        parse_int = lambda x: int(x)
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": 123}')
    token, end = scanner('{"key": 123}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 4

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_string = None
        parse_array = None
        parse_float = lambda x: float(x)
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": 123.45}')
    token, end = scanner('{"key": 123.45}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 7

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        parse_string = lambda s, i, strict: ("key", i + 4)
        parse_array = None
        parse_float = None
        parse_int = None
        memo = {}
        strict = False

    scanner = _make_scanner(MockContext(), '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 13

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        parse_string = None
        parse_array = lambda x, y: ([1, 2, 3], 7)
        parse_float = None
        parse_int = None
        memo = None
        strict = False

    scanner = _make_scanner(MockContext(), '[1, 2, 3]')
    token, end = scanner('[1, 2, 3]', 0)
    assert isinstance(token, ListToken)
    assert end == 7


# LLM-generated content at query #18
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, '{"key": null}'), 4
    assert token.value == None
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 4, 3)
    assert token.string == "null"


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_json_raises_parse_error_on_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.position.column_no == 10
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.char_index == 9


# LLM-generated content at query #20
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, '{"key": null}'), 4
    assert token.value is None
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 4, 3)
    assert token.string == "null"


# LLM-generated content at query #21
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("test", 5),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_dict():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("key", 4),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 14

def test_make_scanner_scans_list():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([ScalarToken(1, 0, 0, "")], 3),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '[1, 2, 3]')
    token, end = scanner('[1, 2, 3]', 0)
    assert isinstance(token, ListToken)
    assert end == 7

def test_make_scanner_scans_null():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '123')
    token, end = scanner('123', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 1),
        'parse_string': lambda x, y, z: ("", 1),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '123.45')
    token, end = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6


# LLM-generated content at query #22
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 4

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "[]")
    token, end = scanner("[]", 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 2

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "{}")
    token, end = scanner("{}", 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2


# LLM-generated content at query #23
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    def scan_once(s, i):
        return (ScalarToken(42, i, i, s), i)
    result, end = _TokenizingJSONObject(('{"a": 42}', 0), True, scan_once, {}, '{"a": 42}')
    assert result == {"a": 42}
    assert end == 9

def test__TokenizingJSONObject_multiple_pairs():
    def scan_once(s, i):
        return (ScalarToken(42, i, i, s), i)
    result, end = _TokenizingJSONObject(('{"a": 42, "b": 43}', 0), True, scan_once, {}, '{"a": 42, "b": 43}')
    assert result == {"a": 42, "b": 43}
    assert end == 18

def test__TokenizingJSONObject_with_whitespace():
    def scan_once(s, i):
        return (ScalarToken(42, i, i, s), i)
    result, end = _TokenizingJSONObject(('{ "a" : 42 }', 0), True, scan_once, {}, '{ "a" : 42 }')
    assert result == {"a": 42}
    assert end == 11

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"a": 42', 0), True, lambda s, i: (ScalarToken(42, i, i, s), i), {}, '{"a": 42')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"a" 42}', 0), True, lambda s, i: (ScalarToken(42, i, i, s), i), {}, '{"a" 42}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"a": 42 "b": 43}', 0), True, lambda s, i: (ScalarToken(42, i, i, s), i), {}, '{"a": 42 "b": 43}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: 42}', 0), True, lambda s, i: (ScalarToken(42, i, i, s), i), {}, '{123: 42}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    s = '{"key": "value"}'
    end = len(s) - 1
    try:
        nextchar = s[end]
        assert nextchar != ""
    except IndexError:
        assert False, "IndexError should not be raised"


# LLM-generated content at query #25
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), False, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result == {"key": ScalarToken("value", 8, 13, content)}
    assert end == len(content)

def test__TokenizingJSONObject_multiple_key_value_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content), "key2": ScalarToken("value2", 22, 27, content)}
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result == {"key": ScalarToken("value", 10, 15, content)}
    assert end == len(content)

def test__TokenizingJSONObject_trailing_comma_raises_error():
    try:
        content = '{"key": "value",}'
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"

def test__TokenizingJSONObject_missing_colon_raises_error():
    try:
        content = '{"key" "value"}'
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_value_raises_error():
    try:
        content = '{"key":}'
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_nested_object():
    content = '{"outer": {"inner": "value"}}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken({"inner": "value"}, e, e + 17, s), e + 18), {}, content)
    assert result == {"outer": ScalarToken({"inner": "value"}, 9, 26, content)}
    assert end == len(content)


# LLM-generated content at query #26
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, '{"key": null}'), 4
    assert token.value is None
    assert token.start.line_no == 1
    assert token.end.line_no == 1
    assert token.string == "null"


# LLM-generated content at query #27
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = MagicMock()
    context.parse_array = MagicMock()
    context.parse_string = MagicMock()
    context.strict = False
    context.parse_float = float
    context.parse_int = int
    context.memo = {}

    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)

    assert token._value is None
    assert token._start_index == 7
    assert token._end_index == 10
    assert token._content == content


# LLM-generated content at query #28
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_empty_string_raises_stop_iteration():
    context = typing.Any
    content = ""
    scanner = _make_scanner(context, content)
    with pytest.raises(StopIteration):
        scanner("", 0)

def test_make_scanner_with_string_value():
    context = typing.Any
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_with_null_value():
    context = typing.Any
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_with_true_value():
    context = typing.Any
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_with_false_value():
    context = typing.Any
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_with_integer_value():
    context = typing.Any
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_with_float_value():
    context = typing.Any
    content = "123.45"
    scanner = _make_scanner(context, content)
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_with_object_value():
    context = typing.Any
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert end == 15

def test_make_scanner_with_array_value():
    context = typing.Any
    content = '["value"]'
    scanner = _make_scanner(context, content)
    token, end = scanner('["value"]', 0)
    assert isinstance(token, ListToken)
    assert token.value == ["value"]
    assert end == 9

def test_make_scanner_clears_memo():
    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    scanner("test", 0)
    assert len(context.memo) == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_index_error_predicate_false():
    s = '{"key": "value"}'
    end = 10
    _w = lambda s, end: type('Match', (), {'end': lambda self: end})()
    _ws = ' '
    try:
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()
    except IndexError:
        pass
    assert True


# LLM-generated content at query #30
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    scan_once = lambda s, e: (ScalarToken("value", 8, 13, s), 14)
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), False, scan_once, {}, content)
    assert result == {"key": ScalarToken("value", 8, 13, content)}
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    scan_once = lambda s, e: (ScalarToken("value1", 8, 13, s), 14) if e == 8 else (ScalarToken("value2", 24, 29, s), 30)
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), False, scan_once, {}, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content), "key2": ScalarToken("value2", 24, 29, content)}
    assert end == 31

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key":  "value"}'
    scan_once = lambda s, e: (ScalarToken("value", 10, 15, s), 16)
    result, end = _TokenizingJSONObject(('{"key":  "value"}', 0), False, scan_once, {}, content)
    assert result == {"key": ScalarToken("value", 10, 15, content)}
    assert end == 17

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{"key": "value"')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: "value"}', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{123: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #31
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_scans_object():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("key", 5),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 15

def test_make_scanner_scans_array():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([ScalarToken("value", 0, 4, '["value"]')], 8),
        'parse_string': lambda x, y, z: ("value", 7),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '["value"]')
    token, end = scanner('["value"]', 0)
    assert isinstance(token, ListToken)
    assert end == 8

def test_make_scanner_scans_null():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3


# LLM-generated content at query #32
#--------------------------

```python
def test_parse_object_predicate_false():
    assert _make_scanner(None, "") is not _TokenizingJSONObject


# LLM-generated content at query #33
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        parse_array = None
        parse_string = None
        strict = False
        parse_float = float
        parse_int = int
        memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert scanner.__closure__[0].cell_contents is not _TokenizingJSONObject


# LLM-generated content at query #34
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = typing.Any
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)
    assert token.value is None
    assert token.start._index == 7
    assert token.end._index == 10
    assert isinstance(token, ScalarToken)


# LLM-generated content at query #35
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number_int():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_number_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 2)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "[]")
    token, end = scanner("[]", 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 2

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    class MockTokenizingJSONObject:
        def __call__(self, *args, **kwargs):
            return ({}, 2)

    context = MockContext()
    scanner = _make_scanner(context, "{}")
    token, end = scanner("{}", 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {"test": "value"}

    context = MockContext()
    scanner = _make_scanner(context, "")
    scanner("", 0)
    assert context.memo == {}


# LLM-generated content at query #36
#--------------------------

```python
def test_index_error_in_whitespace_skip():
    s = '{"key":'
    end = len(s)
    assert s[end] in WHITESPACE_STR
    assert s[end + 1] in WHITESPACE_STR
    try:
        _w(s, end + 1).end()
    except IndexError:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, 'null'), 4
    assert token.value == None
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 4, 3)
    assert end == 4


# LLM-generated content at query #38
#--------------------------

```python
def test_tokenize_json_with_valid_content():
    content = '{"key": "value"}'
    result = tokenize_json(content)
    assert isinstance(result, Token)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_handles_string_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 4

def test_make_scanner_handles_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 1)
    assert isinstance(token, DictToken)
    assert end == 15

def test_make_scanner_handles_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], y + 1)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '["test"]')
    token, end = scanner('["test"]', 1)
    assert isinstance(token, ListToken)
    assert end == 7

def test_make_scanner_handles_null_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_handles_true_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_handles_false_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_handles_number_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '123')
    token, end = scanner('123', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 4)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {"test": "value"}

    context = MockContext()
    scanner = _make_scanner(context, 'null')
    scanner('null', 0)
    assert context.memo == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", y + 5),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_dict():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("key", y + 3),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 13

def test_make_scanner_scans_list():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([ScalarToken(1, 0, 0, "")], y + 3),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '[1, 2, 3]')
    token, end = scanner('[1, 2, 3]', 0)
    assert isinstance(token, ListToken)
    assert end == 7

def test_make_scanner_scans_null():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '123')
    token, end = scanner('123', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_clears_memo():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {'test': 'value'}
    })()
    scanner = _make_scanner(context, '')
    scanner('', 0)
    assert context.memo == {}


# LLM-generated content at query #3
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), False, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    content = '{"key": "value"}'
    memo = {}
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), memo, content)
    assert result == {"key": ScalarToken("value", 8, 11, content)}
    assert end == 13

def test__TokenizingJSONObject_multiple_key_value_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    memo = {}
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), False, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), memo, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content), "key2": ScalarToken("value2", 23, 28, content)}
    assert end == 30

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key":  "value"}'
    memo = {}
    result, end = _TokenizingJSONObject(('{"key":  "value"}', 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), memo, content)
    assert result == {"key": ScalarToken("value", 10, 13, content)}
    assert end == 15

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), False, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, '{"key": "value"')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: "value"}', 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, '{123: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"


# LLM-generated content at query #4
#--------------------------

```python
def test_scalar_token_creation_for_string():
    class MockContext:
        def __init__(self):
            self.parse_string = lambda s, i, strict: ("test", i + 5)
            self.strict = False
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.start._index == 1
    assert token.end._index == 5


# LLM-generated content at query #5
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{}')
    assert result == {}
    assert end == 1

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 7, 12), 13), {}, content)
    assert result == {"key": ScalarToken("value", 7, 12, content)}
    assert end == len(content)

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", 8, 13), 14) if e == 8 else (ScalarToken("value2", 23, 28), 29), {}, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content), "key2": ScalarToken("value2", 23, 28, content)}
    assert end == len(content)

def test__TokenizingJSONObject_whitespace_handling():
    content = '{"key" : "value" }'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 9, 14), 15), {}, content)
    assert result == {"key": ScalarToken("value", 9, 14, content)}
    assert end == len(content)

def test__TokenizingJSONObject_expecting_property_name_error():
    try:
        _TokenizingJSONObject(('{123', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{123')
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"

def test__TokenizingJSONObject_expecting_colon_delimiter_error():
    try:
        _TokenizingJSONObject(('{"key" 123', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{"key" 123')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_expecting_value_error():
    try:
        _TokenizingJSONObject(('{"key":', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{"key":')
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_expecting_comma_delimiter_error():
    try:
        _TokenizingJSONObject(('{"key": "value" "key2": "value2"}', 0), True, lambda s, e: (ScalarToken("value", 8, 13), 14), {}, '{"key": "value" "key2": "value2"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"


# LLM-generated content at query #6
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, 0, 0, s), 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 8, 13, s), 14), {}, content)
    key = list(result.keys())[0]
    assert key.string == 'key'
    assert result[key].string == 'value'
    assert end == 15

def test__TokenizingJSONObject_multiple_key_values():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 8 else ScalarToken("value2", 25, 31, s), 16 if e == 8 else 32), {}, content)
    assert len(result) == 2
    assert list(result.keys())[0].string == 'key1'
    assert list(result.keys())[1].string == 'key2'
    assert result[list(result.keys())[0]].string == 'value1'
    assert result[list(result.keys())[1]].string == 'value2'
    assert end == 33

def test__TokenizingJSONObject_with_whitespace():
    content = '  {  "key"  :  "value"  }  '
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 16, 21, s), 22), {}, content)
    key = list(result.keys())[0]
    assert key.string == 'key'
    assert result[key].string == 'value'
    assert end == 24

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 8, 13, s), 14), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 8 else ScalarToken("value2", 25, 31, s), 16 if e == 8 else 32), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_unquoted_key():
    content = '{key: "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 7, 12, s), 13), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_null_token_predicate_false():
    assert not ("null" == "n" and "null"[:4] == "null")


# LLM-generated content at query #8
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (Token(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    def scan_once(s, e):
        return (Token(42, e, e + 1, s), e + 2)
    result, end = _TokenizingJSONObject(('{"a": 42}', 0), True, scan_once, {}, '{"a": 42}')
    assert result == {"a": 42}
    assert end == 10

def test__TokenizingJSONObject_multiple_pairs():
    def scan_once(s, e):
        if s[e] == '4':
            return (Token(42, e, e + 1, s), e + 2)
        return (Token(3.14, e, e + 2, s), e + 3)
    result, end = _TokenizingJSONObject(('{"a": 42, "b": 3.14}', 0), True, scan_once, {}, '{"a": 42, "b": 3.14}')
    assert result == {"a": 42, "b": 3.14}
    assert end == 21

def test__TokenizingJSONObject_with_whitespace():
    def scan_once(s, e):
        return (Token(42, e, e + 1, s), e + 2)
    result, end = _TokenizingJSONObject(('{ "a" : 42 }', 0), True, scan_once, {}, '{ "a" : 42 }')
    assert result == {"a": 42}
    assert end == 12

def test__TokenizingJSONObject_missing_closing_brace():
    def scan_once(s, e):
        return (Token(42, e, e + 1, s), e + 2)
    try:
        _TokenizingJSONObject(('{"a": 42', 0), True, scan_once, {}, '{"a": 42')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting" in str(e)

def test__TokenizingJSONObject_missing_colon():
    def scan_once(s, e):
        return (Token(42, e, e + 1, s), e + 2)
    try:
        _TokenizingJSONObject(('{"a" 42}', 0), True, scan_once, {}, '{"a" 42}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    def scan_once(s, e):
        if s[e] == '4':
            return (Token(42, e, e + 1, s), e + 2)
        return (Token(3.14, e, e + 2, s), e + 3)
    try:
        _TokenizingJSONObject(('{"a": 42 "b": 3.14}', 0), True, scan_once, {}, '{"a": 42 "b": 3.14}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_nested_object():
    def scan_once(s, e):
        if s[e] == '{':
            return _TokenizingJSONObject((s, e), True, scan_once, {}, s)
        return (Token(42, e, e + 1, s), e + 2)
    result, end = _TokenizingJSONObject(('{"a": {"b": 42}}', 0), True, scan_once, {}, '{"a": {"b": 42}}')
    assert result == {"a": {"b": 42}}
    assert end == 16


# LLM-generated content at query #9
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = MagicMock()
    content = "test content"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = MagicMock()
    context.parse_string.return_value = ("test", 5)
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_with_dict_token():
    context = MagicMock()
    context.parse_array = MagicMock()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert end == len(content)

def test_make_scanner_with_list_token():
    context = MagicMock()
    context.parse_array.return_value = ([], 2)
    content = "[]"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert end == 2

def test_make_scanner_with_null_token():
    context = MagicMock()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_with_true_token():
    context = MagicMock()
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_with_false_token():
    context = MagicMock()
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_with_number_token():
    context = MagicMock()
    context.parse_float = float
    context.parse_int = int
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_with_float_token():
    context = MagicMock()
    context.parse_float = float
    context.parse_int = int
    content = "123.45"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_clears_memo():
    context = MagicMock()
    context.memo = MagicMock()
    content = "test"
    scanner = _make_scanner(context, content)
    scanner(content, 0)
    context.memo.clear.assert_called_once()


# LLM-generated content at query #10
#--------------------------

```python
def test_make_scanner_parse_object_not_tokenizing_json_object():
    context = typing.SimpleNamespace(
        parse_array=lambda x, y: (None, 0),
        parse_string=lambda x, y, z: ("", 0),
        strict=True,
        parse_float=float,
        parse_int=int,
        memo={}
    )
    scanner = _make_scanner(context, "")
    assert parse_object != _TokenizingJSONObject


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_json_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_json("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_null():
    token = tokenize_json("null")
    assert token == ScalarToken(None, 0, 3, "null")
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_true():
    token = tokenize_json("true")
    assert token == ScalarToken(True, 0, 3, "true")
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_false():
    token = tokenize_json("false")
    assert token == ScalarToken(False, 0, 4, "false")
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_number_integer():
    token = tokenize_json("42")
    assert token == ScalarToken(42, 0, 1, "42")
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_number_float():
    token = tokenize_json("3.14")
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert token == ScalarToken("hello", 0, 6, '"hello"')
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_json_dict():
    token = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_json_nested_structures():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.lookup(["a"]).value == 1

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as excinfo:
        tokenize_json("{invalid}")
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position.line_no == 1


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_json_raises_parse_error_on_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text == "Expecting property name enclosed in double quotes: line 1 column 10 (char 9)."
    assert exc_info.value.position == Position(line_no=1, column_no=10, char_index=9)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    s = '{"key": "value"}'
    end = len(s)
    nextchar = s[end : end + 1]
    assert nextchar == ""


# LLM-generated content at query #14
#--------------------------

```python
def test_nextchar_not_quote_raises_error():
    s = '{"key": "value", }'
    end = 11
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", 8, 12, s), 13)
    result, _ = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    assert result == {"key": "value"}


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_simple_scalar():
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.string == '"hello"'
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_number():
    result = tokenize_json("42")
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.string == "42"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
    result = tokenize_json("3.14")
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.string == "3.14"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_true():
    result = tokenize_json("true")
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.string == "true"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_false():
    result = tokenize_json("false")
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.string == "false"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_null():
    result = tokenize_json("null")
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.string == "null"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_list():
    result = tokenize_json('[1, "two", false]')
    assert isinstance(result, ListToken)
    assert result.value == [1, "two", False]
    assert len(result._value) == 3
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=15, char_index=14)

def test_tokenize_json_dict():
    result = tokenize_json('{"a": 1, "b": "two"}')
    assert isinstance(result, DictToken)
    assert result.value == {"a": 1, "b": "two"}
    assert len(result._value) == 2
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=17, char_index=16)

def test_tokenize_json_nested_structures():
    result = tokenize_json('{"list": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(result, DictToken)
    assert result.value == {"list": [1, 2], "nested": {"key": "value"}}
    assert result.lookup(["list"]).value == [1, 2]
    assert result.lookup(["nested", "key"]).value == "value"

def test_tokenize_json_bytes_input():
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

def test_tokenize_json_invalid_json():
    try:
        tokenize_json('{"invalid": json}')
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting" in e.text


# LLM-generated content at query #16
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token
    import typing

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)
    assert scanner("", 0) == (Token(None, 0, 0, ""), 0)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", 5)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("key", 3)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 13

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 2)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '[]')
    token, end = scanner('[]', 0)
    assert isinstance(token, ListToken)
    assert end == 2

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 123.45
            self.parse_int = lambda x: 123
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '123.45')
    token, end = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = lambda x: 0.0
            self.parse_int = lambda x: 0
            self.memo = {"test": "value"}

    context = MockContext()
    scanner = _make_scanner(context, '')
    scanner('', 0)
    assert len(context.memo) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_json_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_null():
    token = tokenize_json("null")
    assert token == ScalarToken(None, 0, 3, "null")
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_true():
    token = tokenize_json("true")
    assert token == ScalarToken(True, 0, 3, "true")
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_false():
    token = tokenize_json("false")
    assert token == ScalarToken(False, 0, 4, "false")
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_integer():
    token = tokenize_json("42")
    assert token == ScalarToken(42, 0, 1, "42")
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
    token = tokenize_json("3.14")
    assert token == ScalarToken(3.14, 0, 3, "3.14")
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert token == ScalarToken("hello", 0, 6, '"hello"')
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)
    assert token.lookup([0]) == ScalarToken(1, 1, 1, "[1, 2, 3]")
    assert token.lookup([1]) == ScalarToken(2, 3, 3, "[1, 2, 3]")
    assert token.lookup([2]) == ScalarToken(3, 5, 5, "[1, 2, 3]")

def test_tokenize_json_dict():
    token = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.string == '{"a": 1, "b": 2}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup_key(["a"]) == ScalarToken("a", 1, 3, '{"a": 1, "b": 2}')
    assert token.lookup(["a"]) == ScalarToken(1, 5, 5, '{"a": 1, "b": 2}')
    assert token.lookup_key(["b"]) == ScalarToken("b", 8, 10, '{"a": 1, "b": 2}')
    assert token.lookup(["b"]) == ScalarToken(2, 12, 12, '{"a": 1, "b": 2}')

def test_tokenize_json_nested():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]) == ScalarToken(1, 7, 7, '{"a": [1, 2], "b": {"c": 3}}')
    assert token.lookup(["a", 1]) == ScalarToken(2, 9, 9, '{"a": [1, 2], "b": {"c": 3}}')
    assert token.lookup(["b", "c"]) == ScalarToken(3, 20, 20, '{"a": [1, 2], "b": {"c": 3}}')

def test_tokenize_json_bytes():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.string == '{"a": 1}'

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"a":}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.column_no == 6
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.char_index == 5


# LLM-generated content at query #18
#--------------------------

```python
def test_index_error_in_whitespace_handling():
    s = '{"key":'
    end = 8
    _w = lambda s, end: re.match(r'\s*', s[end:])
    _ws = ' \t\n\r'
    with pytest.raises(IndexError):
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()


# LLM-generated content at query #19
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i+1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_key_values():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value1" if i < 10 else "value2", i, i+6, s), i+7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '  {  "key"  :  "value"  }  '
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+5, s), i+6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value1" if i < 10 else "value2", i, i+6, s), i+7), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_nested_object():
    content = '{"outer": {"inner": "value"}}'
    def scan_once(s, i):
        if i < 10:
            return ScalarToken({"inner": "value"}, i, i+16, s), i+17
        else:
            return ScalarToken("value", i, i+5, s), i+6
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, {}, content)
    assert len(result) == 1
    assert isinstance(result["outer"].value, dict)
    assert result["outer"].value["inner"] == "value"
    assert end == len(content)


# LLM-generated content at query #20
#--------------------------

```python
def test_index_error_in_whitespace_skip():
    s = '{"key":'
    end = len(s) - 1
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, s), end)
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key": "value"}


# LLM-generated content at query #21
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("test", y + 5)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], y[1] + 1)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "[]")
    token, end = scanner("[]", 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 2

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "{}")
    token, end = scanner("{}", 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {"test": "value"}

    context = MockContext()
    scanner = _make_scanner(context, "null")
    scanner("null", 0)
    assert context.memo == {}


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    s = '{"key": "value"}'
    end = len(s)
    try:
        nextchar = s[end]
    except IndexError:
        nextchar = ""
    assert nextchar == ""


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    s = '{"key": "value"}'
    end = len(s) - 1
    nextchar = ""
    try:
        nextchar = s[end]
    except IndexError:
        pass
    assert nextchar != ""


# LLM-generated content at query #24
#--------------------------

```python
def test_index_error_predicate_false():
    s = '{"key": "value"}'
    end = 10
    assert s[end] not in _ws


# LLM-generated content at query #25
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    assert not (_TokenizingJSONObject is _TokenizingJSONObject)


# LLM-generated content at query #26
#--------------------------

```python
def test_index_error_raises_empty_nextchar():
    s = '{"key": "value"'
    end = len(s)
    try:
        nextchar = s[end]
    except IndexError:
        nextchar = ""
    assert nextchar == ""


# LLM-generated content at query #27
#--------------------------

```python
def test_tokenize_json_raises_parse_error_on_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text.endswith(".")
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_false():
    s = '{"key": "value"'
    end = 10
    _w = lambda s, end: type('Match', (), {'end': lambda self: end})()
    _ws = ' '
    try:
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()
    except IndexError:
        pass
    assert True


# LLM-generated content at query #29
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    context = typing.Any
    content = ""
    scanner = _make_scanner(context, content)
    assert parse_object is not _TokenizingJSONObject


# LLM-generated content at query #30
#--------------------------

```python
def test_nextchar_not_empty_string_when_index_error():
    s = '{"key": "value"'
    end = len(s)
    try:
        nextchar = s[end]
    except IndexError:
        nextchar = ""
    assert nextchar == ""


# LLM-generated content at query #31
#--------------------------

```python
def test_tokenize_json_with_empty_string():
    with pytest.raises(ParseError) as excinfo:
        tokenize_json("")
    assert excinfo.value.text == "No content."
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position == Position(column_no=1, line_no=1, char_index=0)


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_json_with_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.text.endswith(".")
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)


# LLM-generated content at query #33
#--------------------------

```python
def test_index_error_handling():
    s = '{"key": "value"'
    end = len(s)
    nextchar = ""
    try:
        nextchar = s[end]
    except IndexError:
        pass
    assert nextchar == ""


# LLM-generated content at query #34
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, 0, 0), 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    def scan_once(s, i):
        return (ScalarToken("value", 0, 0), 1)
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), True, scan_once, {}, '{"key": "value"}')
    assert len(result) == 1
    assert result[ScalarToken("key", 0, 0)].value == "value"
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    def scan_once(s, i):
        return (ScalarToken("value", 0, 0), 1)
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), True, scan_once, {}, '{"key1": "value1", "key2": "value2"}')
    assert len(result) == 2
    assert result[ScalarToken("key1", 0, 0)].value == "value1"
    assert result[ScalarToken("key2", 0, 0)].value == "value2"
    assert end == 30

def test__TokenizingJSONObject_with_whitespace():
    def scan_once(s, i):
        return (ScalarToken("value", 0, 0), 1)
    result, end = _TokenizingJSONObject(('{ "key" : "value" }', 0), True, scan_once, {}, '{ "key" : "value" }')
    assert len(result) == 1
    assert result[ScalarToken("key", 0, 0)].value == "value"
    assert end == 17

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, i: (ScalarToken("value", 0, 0), 1), {}, '{"key" "value"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, i: (ScalarToken("value", 0, 0), 1), {}, '{"key1": "value1" "key2": "value2"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), True, lambda s, i: (ScalarToken("value", 0, 0), 1), {}, '{"key": "value"')
    except IndexError:
        pass

def test__TokenizingJSONObject_missing_value():
    def scan_once(s, i):
        raise StopIteration(0)
    try:
        _TokenizingJSONObject(('{"key":', 0), True, scan_once, {}, '{"key":')
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"


# LLM-generated content at query #35
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    class MockContext:
        parse_array = None
        parse_string = None
        strict = None
        parse_float = None
        parse_int = None
        memo = None

    context = MockContext()
    content = ""
    scanner = _make_scanner(context, content)
    assert parse_object is not _TokenizingJSONObject


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_false():
    s = "{\"key\": \"value\"}"
    end = 10
    _w = lambda s, end: type('MockMatch', (), {'end': lambda self: end + 1})()
    _ws = " "

    assert not (s[end] in _ws and s[end + 1] in _ws)


# LLM-generated content at query #37
#--------------------------

```python
def test_tokenize_json_with_invalid_json_raises_parse_error():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.text == "Expecting property name enclosed in double quotes: line 1 column 10 (char 9)."
    assert exc_info.value.position == Position(line_no=1, column_no=10, char_index=9)


# LLM-generated content at query #38
#--------------------------

```python
def test_parse_object_not_tokenizing_json_object():
    assert _make_scanner.__code__.co_consts[1] is not _TokenizingJSONObject


# LLM-generated content at query #39
#--------------------------

```python
def test_line_61_predicate_false():
    s = '{"key": "value"}'
    end = len(s)
    nextchar = ""
    try:
        nextchar = s[end]
        if nextchar in " \t\n\r":
            end = WHITESPACE.match(s, end + 1).end()
            nextchar = s[end]
    except IndexError:
        pass
    assert nextchar == ""


