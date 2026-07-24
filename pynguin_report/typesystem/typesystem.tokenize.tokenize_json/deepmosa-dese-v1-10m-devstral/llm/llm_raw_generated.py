####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    scanner = _make_scanner(context, '{"key": "test"}')
    token, end = scanner('"test"', 0)
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
        'parse_array': lambda x, y: ([ScalarToken(1, 0, 0, "")], y + 2),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '[1]')
    token, end = scanner('[1]', 0)
    assert isinstance(token, ListToken)
    assert end == 2

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

def test_make_scanner_scans_float():
    context = type('Context', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '123.45')
    token, end = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6


# LLM-generated content at query #2
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)
    result = scanner("", 0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Token)
    assert isinstance(result[1], int)


# LLM-generated content at query #3
#--------------------------

```python
def test_make_scanner_parse_object_not_tokenizing_json_object():
    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert parse_object != _TokenizingJSONObject


# LLM-generated content at query #4
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, 0, 0), 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    def mock_scan_once(s, e):
        return (ScalarToken(42, e, e), e + 1)
    result, end = _TokenizingJSONObject(('{"key": 42}', 0), True, mock_scan_once, {}, '{"key": 42}')
    assert result == {"key": ScalarToken(42, 3, 3, '{"key": 42}')}
    assert end == 10

def test__TokenizingJSONObject_multiple_pairs():
    def mock_scan_once(s, e):
        if e == 5:
            return (ScalarToken(42, e, e), e + 1)
        return (ScalarToken("value", e, e + 4), e + 5)
    result, end = _TokenizingJSONObject(('{"key": 42, "key2": "value"}', 0), True, mock_scan_once, {}, '{"key": 42, "key2": "value"}')
    assert len(result) == 2
    assert result["key"] == ScalarToken(42, 5, 5, '{"key": 42, "key2": "value"}')
    assert result["key2"] == ScalarToken("value", 17, 21, '{"key": 42, "key2": "value"}')
    assert end == 28

def test__TokenizingJSONObject_with_whitespace():
    def mock_scan_once(s, e):
        return (ScalarToken(42, e, e), e + 1)
    result, end = _TokenizingJSONObject(('{ "key" : 42 }', 0), True, mock_scan_once, {}, '{ "key" : 42 }')
    assert result == {"key": ScalarToken(42, 8, 8, '{ "key" : 42 }')}
    assert end == 13

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": 42', 0), True, lambda s, e: (ScalarToken(42, 0, 0), 1), {}, '{"key": 42')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting" in str(e)

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" 42}', 0), True, lambda s, e: (ScalarToken(42, 0, 0), 1), {}, '{"key" 42}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key": 42 "key2": 43}', 0), True, lambda s, e: (ScalarToken(42, 0, 0), 1), {}, '{"key": 42 "key2": 43}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: 42}', 0), True, lambda s, e: (ScalarToken(42, 0, 0), 1), {}, '{123: 42}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #5
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

def test_tokenize_json_integer():
    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
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
    assert token.end == Position(line_no=1, column_no=17, char_index=16)
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup_key(["b"]).value == "b"

def test_tokenize_json_nested():
    token = tokenize_json('{"a": {"b": [1, 2]}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": [1, 2]}}
    assert token.lookup(["a", "b", 0]).value == 1
    assert token.lookup(["a", "b", 1]).value == 2

def test_tokenize_json_bytes():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("{invalid}")
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #6
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, '{"key": null}'), 4
    assert token._value is None
    assert token._start_index == 0
    assert token._end_index == 3
    assert token._content == '{"key": null}'
    assert end == 4


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_false():
    s = '{"key": "value"}'
    end = 10
    _w = lambda s, end: type('match', (), {'end': lambda self: end})()
    _ws = ' \t\n\r'
    with pytest.raises(IndexError):
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()


# LLM-generated content at query #8
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i), {}, '{}')
    assert result == ({}, 2)

def test__TokenizingJSONObject_single_pair():
    scan_once = lambda s, i: (ScalarToken('value', i, i+4, s), i+5)
    result = _TokenizingJSONObject(('{"key": "value"}', 0), True, scan_once, {}, '{"key": "value"}')
    assert len(result[0]) == 1
    assert result[0]['key'].value == 'value'
    assert result[1] == 15

def test__TokenizingJSONObject_multiple_pairs():
    scan_once = lambda s, i: (ScalarToken('value', i, i+4, s), i+5)
    result = _TokenizingJSONObject(('{"k1": "v1", "k2": "v2"}', 0), True, scan_once, {}, '{"k1": "v1", "k2": "v2"}')
    assert len(result[0]) == 2
    assert result[0]['k1'].value == 'v1'
    assert result[0]['k2'].value == 'v2'
    assert result[1] == 24

def test__TokenizingJSONObject_with_whitespace():
    scan_once = lambda s, i: (ScalarToken('value', i, i+4, s), i+5)
    result = _TokenizingJSONObject(('{ "key" : "value" }', 0), True, scan_once, {}, '{ "key" : "value" }')
    assert len(result[0]) == 1
    assert result[0]['key'].value == 'value'
    assert result[1] == 18

def test__TokenizingJSONObject_nested_object():
    scan_once = lambda s, i: (ScalarToken({'nested': 'value'}, i, i+16, s), i+17)
    result = _TokenizingJSONObject(('{"outer": {"nested": "value"}}', 0), True, scan_once, {}, '{"outer": {"nested": "value"}}')
    assert len(result[0]) == 1
    assert result[0]['outer'].value == {'nested': 'value'}
    assert result[1] == 28


# LLM-generated content at query #9
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
    assert token.string == "null"


# LLM-generated content at query #10
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, end: (ScalarToken(None, 0, 0), 1), {}, '{}')
    assert result == {}
    assert end == 1

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 12, content)
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), True, lambda s, end: (value_token, 13), {}, content)
    assert result == {"key": "value"}
    assert end == 13

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    key1_token = ScalarToken("key1", 1, 4, content)
    value1_token = ScalarToken("value1", 8, 13, content)
    key2_token = ScalarToken("key2", 17, 20, content)
    value2_token = ScalarToken("value2", 24, 28, content)
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), True, lambda s, end: (value1_token if end == 8 else value2_token, 14 if end == 8 else 29), {}, content)
    assert result == {"key1": "value1", "key2": "value2"}
    assert end == 29

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key":  "value"}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 10, 14, content)
    result, end = _TokenizingJSONObject(('{"key":  "value"}', 0), True, lambda s, end: (value_token, 15), {}, content)
    assert result == {"key": "value"}
    assert end == 15

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, end: (ScalarToken("value", 0, 0), 1), {}, '{"key" "value"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, end: (ScalarToken("value1", 0, 0), 1), {}, '{"key1": "value1" "key2": "value2"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), True, lambda s, end: (ScalarToken("value", 0, 0), 1), {}, '{"key": "value"')
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: "value"}', 0), True, lambda s, end: (ScalarToken("value", 0, 0), 1), {}, '{123: "value"}')
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"


# LLM-generated content at query #11
#--------------------------

```python
def test_null_token_creation():
    token, end = ScalarToken(None, 0, 3, 'null'), 4
    assert token.value == None
    assert token.start == 0
    assert token.end == 3
    assert token.string == 'null'


# LLM-generated content at query #12
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, scan_once, {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    s = '{"key": "value"}'
    content = s
    scan_once = lambda s, end: (ScalarToken("value", 8, 13, content), 14)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    content = s
    scan_once = lambda s, end: (ScalarToken("value1", 9, 15, content), 16) if end == 8 else (ScalarToken("value2", 25, 31, content), 32)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == 33

def test__TokenizingJSONObject_with_whitespace():
    s = '  { "key" : "value" }  '
    content = s
    scan_once = lambda s, end: (ScalarToken("value", 14, 19, content), 20)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == 22

def test__TokenizingJSONObject_missing_colon():
    s = '{"key" "value"}'
    content = s
    try:
        _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    s = '{"key1": "value1" "key2": "value2"}'
    content = s
    try:
        _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    s = '{"key": "value"'
    content = s
    try:
        _TokenizingJSONObject((s, 0), True, scan_once, {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 8, 13, s), 14), {}, content)
    assert result == {"key": ScalarToken("value", 8, 13, content)}
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 9 else ScalarToken("value2", 25, 31, s), e + 7), {}, content)
    assert result == {"key1": ScalarToken("value1", 9, 15, content), "key2": ScalarToken("value2", 25, 31, content)}
    assert end == 33

def test__TokenizingJSONObject_whitespace_handling():
    content = '{"key" : "value" }'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", 10, 15, s), 16), {}, content)
    assert result == {"key": ScalarToken("value", 10, 15, content)}
    assert end == 17

def test__TokenizingJSONObject_missing_quotes_raises_error():
    try:
        _TokenizingJSONObject(('{key: "value"}', 0), True, lambda s, e: (ScalarToken("value", 7, 12, s), 13), {}, '{key: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test__TokenizingJSONObject_missing_colon_raises_error():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, e: (ScalarToken("value", 8, 13, s), 14), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_value_raises_error():
    try:
        _TokenizingJSONObject(('{"key":', 0), True, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{"key":')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test__TokenizingJSONObject_missing_comma_raises_error():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 9 else ScalarToken("value2", 25, 31, s), e + 7), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    assert not (nextchar != '"' for nextchar in ['"', '}'])


# LLM-generated content at query #15
#--------------------------

```python
def test_line_39_predicate_false():
    s = '{"key" : value}'
    end = 7
    assert s[end : end + 1] != ":"


# LLM-generated content at query #16
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
    result = tokenize_json("null")
    assert result == ScalarToken(None, 0, 3, "null")
    assert result.value is None
    assert result.string == "null"
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_boolean():
    true_token = tokenize_json("true")
    assert true_token == ScalarToken(True, 0, 3, "true")
    assert true_token.value is True
    assert true_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert true_token.end == Position(line_no=1, column_no=4, char_index=3)

    false_token = tokenize_json("false")
    assert false_token == ScalarToken(False, 0, 4, "false")
    assert false_token.value is False
    assert false_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert false_token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_number():
    int_token = tokenize_json("42")
    assert int_token == ScalarToken(42, 0, 1, "42")
    assert int_token.value == 42

    float_token = tokenize_json("3.14")
    assert float_token == ScalarToken(3.14, 0, 3, "3.14")
    assert float_token.value == 3.14

def test_tokenize_json_string():
    result = tokenize_json('"hello"')
    assert result == ScalarToken("hello", 0, 6, '"hello"')
    assert result.value == "hello"
    assert result.string == '"hello"'

def test_tokenize_json_list():
    result = tokenize_json("[1, 2, 3]")
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=8, char_index=7)
    assert result.lookup([0]).value == 1
    assert result.lookup([1]).value == 2
    assert result.lookup([2]).value == 3

def test_tokenize_json_dict():
    result = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(result, DictToken)
    assert result.value == {"a": 1, "b": 2}
    assert result.start == Position(line_no=1, column_no=1, char_index=0)
    assert result.end == Position(line_no=1, column_no=14, char_index=13)
    assert result.lookup_key(["a"]).value == "a"
    assert result.lookup(["a"]).value == 1
    assert result.lookup(["b"]).value == 2

def test_tokenize_json_nested_structures():
    result = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(result, DictToken)
    assert result.value == {"a": [1, 2], "b": {"c": 3}}
    assert result.lookup(["a", 0]).value == 1
    assert result.lookup(["a", 1]).value == 2
    assert result.lookup(["b", "c"]).value == 3

def test_tokenize_json_bytes_input():
    result = tokenize_json(b'{"a": 1}')
    assert isinstance(result, DictToken)
    assert result.value == {"a": 1}

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting" in e.text


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)

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
    assert true_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert true_token.end == Position(line_no=1, column_no=4, char_index=3)

    false_token = tokenize_json("false")
    assert false_token == ScalarToken(False, 0, 4, "false")
    assert false_token.value is False
    assert false_token.start == Position(line_no=1, column_no=1, char_index=0)
    assert false_token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_number():
    int_token = tokenize_json("42")
    assert int_token == ScalarToken(42, 0, 1, "42")
    assert int_token.value == 42

    float_token = tokenize_json("3.14")
    assert float_token == ScalarToken(3.14, 0, 3, "3.14")
    assert float_token.value == 3.14

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert token == ScalarToken("hello", 0, 6, '"hello"')
    assert token.value == "hello"
    assert token.string == '"hello"'

def test_tokenize_json_array():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)
    assert token.lookup([0]).value == 1
    assert token.lookup([1]).value == 2
    assert token.lookup([2]).value == 3

def test_tokenize_json_object():
    token = tokenize_json('{"a": 1, "b": 2}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=15, char_index=14)
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup(["b"]).value == 2

def test_tokenize_json_nested_structures():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}

def test_tokenize_json_whitespace():
    token = tokenize_json("  {  \"a\"  :  1  }  ")
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}


# LLM-generated content at query #18
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = typing.Any
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_with_null_token():
    context = typing.Any
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_with_true_token():
    context = typing.Any
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_with_false_token():
    context = typing.Any
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_with_number_token():
    context = typing.Any
    content = '123'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.string == '123'
    assert end == 3

def test_make_scanner_with_list_token():
    context = typing.Any
    content = '[]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_with_dict_token():
    context = typing.Any
    content = '{}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_clears_memo():
    context = typing.Any
    content = 'null'
    scanner = _make_scanner(context, content)
    memo = context.memo
    memo.add('test')
    token, end = scanner(content, 0)
    assert len(memo) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_line_61_predicate_false():
    s = '{"key": "value"'
    end = len(s) - 1
    try:
        nextchar = s[end]
        assert False, "IndexError not raised"
    except IndexError:
        nextchar = ""
        assert nextchar == ""


# LLM-generated content at query #20
#--------------------------

```python
def test_IndexError_handling_in_whitespace_skipping():
    s = '{"key": value'
    end = len(s) - 1  # Position right before the end of the string
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken(None, 0, 0, s), end)

    # Mock the scanstring function to return a key and end position
    def mock_scanstring(s, end, strict):
        return "key", end + 1

    # Mock the WHITESPACE.match function to return a match object with end() method
    class MockMatch:
        def end(self):
            return len(s)

    WHITESPACE = type('MockPattern', (), {'match': lambda s, end: MockMatch()})
    WHITESPACE_STR = ' \t\n\r'

    # Call the function and verify it doesn't raise an IndexError
    result, end_pos = _TokenizingJSONObject(
        (s, 0), False, scan_once, memo, content, WHITESPACE.match, WHITESPACE_STR
    )

    # The function should handle the IndexError gracefully and return the expected result
    assert result == {"key": ScalarToken(None, 0, 0, s)}


# LLM-generated content at query #21
#--------------------------

```python
def test_index_error_handling_in_whitespace_skipping():
    s = '{"key": value'
    end = len(s) - 1
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken(None, end, end, s), end + 1)
    result, new_end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert new_end == len(s)


# LLM-generated content at query #22
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

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

def test_make_scanner_handles_string_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 1)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_handles_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 15

def test_make_scanner_handles_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 2),
        'parse_string': lambda x, y, z: ("test", 6),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '[]')
    token, end = scanner('[1, 2, 3]', 0)
    assert isinstance(token, ListToken)
    assert end == 2

def test_make_scanner_handles_null_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
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

def test_make_scanner_handles_true_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
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

def test_make_scanner_handles_false_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
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

def test_make_scanner_handles_number_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
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

def test_make_scanner_handles_float_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", 6),
        'parse_float': float,
        'parse_int': int,
        'strict': False,
        'memo': {}
    })()
    scanner = _make_scanner(context, '123.45')
    token, end = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6


# LLM-generated content at query #23
#--------------------------

```python
def test_index_error_in_whitespace_optimization():
    s = '{"key": value'
    end = len(s) - 1
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, s), end + 1)
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key": ScalarToken("value", len(s) - 1, len(s) - 1, s)}


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_json_with_valid_content_does_not_raise_exception():
    assert tokenize_json('{"key": "value"}') is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_index_error_handling_in_tokenizing_json_object():
    s = '{"key": "value"}'
    end = len(s)
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, content), end)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
    assert result == {"key": "value"}
    assert end == len(s)


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
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

    context = MockContext()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("test", 6)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

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
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

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
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

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
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

    context = MockContext()
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import re

    class MockContext:
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: float(x)
        parse_int = lambda self, x: int(x)
        memo = {}
        strict = False

    context = MockContext()
    NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    context.match_number = NUMBER_RE.match
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import re

    class MockContext:
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: float(x)
        parse_int = lambda self, x: int(x)
        memo = {}
        strict = False

    context = MockContext()
    NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    context.match_number = NUMBER_RE.match
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        parse_array = lambda self, x, y: ([], 2)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

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
        parse_array = lambda self, x, y: ([], 0)
        parse_string = lambda self, x, y, z: ("", 0)
        parse_float = lambda self, x: 0.0
        parse_int = lambda self, x: 0
        memo = {}
        strict = False

    context = MockContext()
    scanner = _make_scanner(context, "{}")
    token, end = scanner("{}", 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_make_scanner_returns_callable():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
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
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    token, end = scanner('{"test": "value"}', 1)
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
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": null}')
    token, end = scanner('{"test": null}', 9)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 13

def test_make_scanner_scans_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": true}')
    token, end = scanner('{"test": true}', 9)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 13

def test_make_scanner_scans_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": false}')
    token, end = scanner('{"test": false}', 9)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 14

def test_make_scanner_scans_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": 123}')
    token, end = scanner('{"test": 123}', 9)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 12

def test_make_scanner_scans_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": 123.45}')
    token, end = scanner('{"test": 123.45}', 9)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 14

def test_make_scanner_scans_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], x[1] + 2)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": []}')
    token, end = scanner('{"test": []}', 9)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 11

def test_make_scanner_scans_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": {}}')
    token, end = scanner('{"test": {}}', 9)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 11

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner

    class MockContext:
        def __init__(self):
            self.parse_array = lambda x, y: ([], 0)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {"test": "value"}

    context = MockContext()
    scanner = _make_scanner(context, '{"test": "value"}')
    scanner('{"test": "value"}', 0)
    assert context.memo == {}


# LLM-generated content at query #3
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), {}, content)
    assert len(result) == 1
    assert result["key"].string == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_key_values():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value1", i, i+6, s), i+7), {}, content)
    assert len(result) == 2
    assert result["key1"].string == "value1"
    assert end == len(content)

def test__TokenizingJSONObject_whitespace_handling():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), {}, content)
    assert len(result) == 1
    assert result["key"].string == "value"
    assert end == len(content)

def test__TokenizingJSONObject_missing_quotes_raises_error():
    try:
        _TokenizingJSONObject(('{key: "value"}', 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), {}, '{key: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_colon_raises_error():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_missing_comma_raises_error():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, i: (ScalarToken("value1", i, i+6, s), i+7), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test__TokenizingJSONObject_nested_objects():
    content = '{"outer": {"inner": "value"}}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), {}, content)
    assert len(result) == 1
    assert result["outer"].string == '{"inner": "value"}'
    assert end == len(content)


# LLM-generated content at query #4
#--------------------------

```python
def test_index_error_raises_stop_iteration():
    context = typing.Any
    content = ""
    scanner = _make_scanner(context, content)
    with pytest.raises(StopIteration) as exc_info:
        scanner("", 0)
    assert exc_info.value.args[0] == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_IndexError_handling_in_whitespace_skipping():
    s = '{"key": "value"'
    end = len(s)
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, content), end)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
    assert result == {"key": "value"}


# LLM-generated content at query #6
#--------------------------

```python
def test_index_error_in_whitespace_handling():
    s = '{"key":'
    end = len(s) - 1
    content = s
    key = ScalarToken("key", 1, 4, content)
    pairs = [(key, None)]
    memo = {"key": "key"}

    # Simulate the state where end is at the last character of the string
    # and the next operation would cause an IndexError
    with pytest.raises(IndexError):
        _TokenizingJSONObject((s, end), False, lambda s, end: (None, end), memo, content)


# LLM-generated content at query #7
#--------------------------

```python
def test_parse_object_is_not_tokenizing_json_object():
    assert _make_scanner(None, "") is not _TokenizingJSONObject


# LLM-generated content at query #8
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    s = '{}'
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken(None, 0, 0, s), end), {}, s)
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    s = '{"key": "value"}'
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value", 8, 13, s), 14), {}, s)
    assert result == {"key": ScalarToken("value", 8, 13, s)}
    assert end == 15

def test__TokenizingJSONObject_multiple_key_value_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value1", 9, 15, s), 16) if end == 9 else (ScalarToken("value2", 25, 31, s), 32), {}, s)
    assert result == {"key1": ScalarToken("value1", 9, 15, s), "key2": ScalarToken("value2", 25, 31, s)}
    assert end == 33

def test__TokenizingJSONObject_with_whitespace():
    s = '  {  "key"  :  "value"  }  '
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value", 14, 19, s), 20), {}, s)
    assert result == {"key": ScalarToken("value", 14, 19, s)}
    assert end == 22

def test__TokenizingJSONObject_missing_closing_brace():
    s = '{"key": "value"'
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value", 8, 13, s), 14), {}, s)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_colon():
    s = '{"key" "value"}'
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value", 8, 13, s), 14), {}, s)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    s = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, end: (ScalarToken("value1", 9, 15, s), 16) if end == 9 else (ScalarToken("value2", 25, 31, s), 32), {}, s)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_nested_object():
    s = '{"outer": {"inner": "value"}}'
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, end: _TokenizingJSONObject((s, end), True, lambda s, end: (ScalarToken("value", 20, 25, s), 26), {}, s) if s[end] == '{' else (ScalarToken("value", 20, 25, s), 26), {}, s)
    assert result == {"outer": {"inner": ScalarToken("value", 20, 25, s)}}
    assert end == 27


# LLM-generated content at query #9
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    s = '{"key": "value"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), memo, content)
    assert result == {"key": ScalarToken("value", 8, 12, content)}
    assert end == len(s)

def test__TokenizingJSONObject_multiple_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s), e + 7), memo, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content)}
    assert end == len(s)


# LLM-generated content at query #10
#--------------------------

```python
def test_index_error_raises_empty_nextchar():
    s = '{"key": "value"'
    end = len(s)
    nextchar = ""
    try:
        nextchar = s[end]
    except IndexError:
        pass
    assert nextchar == ""


# LLM-generated content at query #11
#--------------------------

```python
def test_line_72_predicate_true():
    s = '{"key1": "value1", "key2": "value2"}'
    end = s.find('"key2"') + len('"key2"')
    nextchar = s[end : end + 1]
    assert nextchar == '"'


# LLM-generated content at query #12
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

def test_tokenize_json_number_integer():
    token = tokenize_json("42")
    assert token == ScalarToken(42, 0, 1, "42")
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_number_float():
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
    assert token.lookup(["a"]) == ScalarToken(1, 6, 6, '{"a": 1, "b": 2}')
    assert token.lookup(["b"]) == ScalarToken(2, 13, 13, '{"a": 1, "b": 2}')
    assert token.lookup_key(["a"]) == ScalarToken("a", 1, 3, '{"a": 1, "b": 2}')
    assert token.lookup_key(["b"]) == ScalarToken("b", 9, 11, '{"a": 1, "b": 2}')

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]) == ScalarToken(1, 7, 7, '{"a": [1, 2], "b": {"c": 3}}')
    assert token.lookup(["a", 1]) == ScalarToken(2, 9, 9, '{"a": [1, 2], "b": {"c": 3}}')
    assert token.lookup(["b", "c"]) == ScalarToken(3, 20, 20, '{"a": [1, 2], "b": {"c": 3}}')
    assert token.lookup_key(["b", "c"]) == ScalarToken("c", 18, 20, '{"a": [1, 2], "b": {"c": 3}}')

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.string == '{"a": 1}'

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"a": 1,}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 8
    assert exc_info.value.position.char_index == 7


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_json_raises_parse_error_on_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": json}')
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #14
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
    assert token.lookup_key(["a"]).value == "a"
    assert token.lookup(["a"]).value == 1
    assert token.lookup_key(["b"]).value == "b"
    assert token.lookup(["b"]).value == 2

def test_tokenize_json_nested_structures():
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": 3}}
    assert token.lookup(["a", 0]).value == 1
    assert token.lookup(["a", 1]).value == 2
    assert token.lookup(["b", "c"]).value == 3

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"a": 1}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"a": 1,}')
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #15
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i+1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    memo = {}
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    memo = {}
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+6, s), i+7), memo, content)
    assert len(result) == 2
    assert result["key1"].value == "value"
    assert result["key2"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_whitespace_handling():
    content = '{"key" : "value"}'
    memo = {}
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), memo, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_nested_object():
    content = '{"outer": {"inner": "value"}}'
    memo = {}
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, i: (ScalarToken("value", i, i+4, s), i+5), memo, content)
    assert len(result) == 1
    assert result["outer"].value == {"inner": "value"}
    assert end == len(content)


# LLM-generated content at query #16
#--------------------------

```python
def test_parse_object_is_not_tokenizing_json_object():
    assert _make_scanner(None, "") is not _TokenizingJSONObject


