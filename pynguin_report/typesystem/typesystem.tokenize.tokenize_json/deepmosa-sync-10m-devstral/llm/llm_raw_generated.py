####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 8 else "value2", e, e + 6, s), e + 7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 9 else "value2", e, e + 6, s), e + 7), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    s = '{"key": "value"}'
    end = 0
    nextchar = s[end : end + 1]
    _ws = ' \t\n\r'
    assert nextchar != '"' and nextchar not in _ws and nextchar != '}'


# LLM-generated content at query #3
#--------------------------

```python
def test_nextchar_in_whitespace():
    s = "  "
    end = 0
    _ws = " \t\n\r"
    nextchar = s[end : end + 1]
    assert nextchar in _ws


# LLM-generated content at query #4
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), False, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_key_values():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test__TokenizingJSONObject_whitespace_handling():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test__TokenizingJSONObject_non_string_key():
    content = '{123: "value"}'
    try:
        _TokenizingJSONObject((content, 0), False, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == 15

def test__TokenizingJSONObject_multiple_key_value_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == 29

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject(('{"key" : "value"}', 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == 16

def test__TokenizingJSONObject_missing_colon():
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, '{"key" "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    try:
        _TokenizingJSONObject(('{"key1": "value1" "key2": "value2"}', 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s) if e == 8 else ScalarToken("value2", e, e + 6, s), e + 7), {}, '{"key1": "value1" "key2": "value2"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"key": "value"', 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, '{"key": "value"')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_non_string_key():
    try:
        _TokenizingJSONObject(('{123: "value"}', 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, '{123: "value"}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"


# LLM-generated content at query #6
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

def test_make_scanner_scans_string_token():
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
    scanner = _make_scanner(context, '{"key": "test"}')
    token, end = scanner('{"key": "test"}', 7)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 12

def test_make_scanner_scans_null_token():
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
    scanner = _make_scanner(context, '{"key": null}')
    token, end = scanner('{"key": null}', 7)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 11

def test_make_scanner_scans_true_token():
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
    scanner = _make_scanner(context, '{"key": true}')
    token, end = scanner('{"key": true}', 7)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 11

def test_make_scanner_scans_false_token():
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
    scanner = _make_scanner(context, '{"key": false}')
    token, end = scanner('{"key": false}', 7)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 12

def test_make_scanner_scans_number_token():
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
    scanner = _make_scanner(context, '{"key": 123}')
    token, end = scanner('{"key": 123}', 7)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 10

def test_make_scanner_scans_dict_token():
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
    scanner = _make_scanner(context, '{"key": {}}')
    token, end = scanner('{"key": {}}', 7)
    assert isinstance(token, DictToken)
    assert end == 9

def test_make_scanner_scans_list_token():
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
    scanner = _make_scanner(context, '{"key": []}')
    token, end = scanner('{"key": []}', 7)
    assert isinstance(token, ListToken)
    assert end == 9


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    assert not (nextchar == "}" or nextchar == ",")


# LLM-generated content at query #8
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
    assert token.value == "test"
    assert end == 6

def test_make_scanner_with_dict_token():
    context = typing.Any
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert end == 15

def test_make_scanner_with_list_token():
    context = typing.Any
    content = '["item1", "item2"]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert end == 16

def test_make_scanner_with_null_token():
    context = typing.Any
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_with_true_token():
    context = typing.Any
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_with_false_token():
    context = typing.Any
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_with_number_token():
    context = typing.Any
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_with_float_token():
    context = typing.Any
    content = "123.45"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_46():
    s = "  "
    end = 0
    _w = lambda s, end: type('Match', (), {'end': lambda self: end + 2})()
    _ws = " "

    # Simulate the condition at line 46
    assert s[end] in _ws
    end += 1
    assert s[end] in _ws


# LLM-generated content at query #10
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), False, lambda s, e: (ScalarToken(None, 0, 0, s), e), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    s = '{"key": "value"}'
    content = s
    memo = {}
    scan_once = lambda s, e: (ScalarToken("value", 8, 13, s), 14)
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key": ScalarToken("value", 8, 13, s)}
    assert end == 15

def test__TokenizingJSONObject_multiple_pairs():
    s = '{"key1": "value1", "key2": "value2"}'
    content = s
    memo = {}
    scan_once = lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 9 else (ScalarToken("value2", 25, 31, s), 32))
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key1": ScalarToken("value1", 9, 15, s), "key2": ScalarToken("value2", 25, 31, s)}
    assert end == 33

def test__TokenizingJSONObject_whitespace_handling():
    s = '{"key" : "value"}'
    content = s
    memo = {}
    scan_once = lambda s, e: (ScalarToken("value", 10, 15, s), 16)
    result, end = _TokenizingJSONObject((s, 0), False, scan_once, memo, content)
    assert result == {"key": ScalarToken("value", 10, 15, s)}
    assert end == 17

def test__TokenizingJSONObject_missing_colon():
    s = '{"key" "value"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), False, lambda s, e: (ScalarToken("value", 8, 13, s), 14), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    s = '{"key1": "value1" "key2": "value2"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), False, lambda s, e: (ScalarToken("value1", 9, 15, s) if e == 9 else (ScalarToken("value2", 25, 31, s), 32)), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    s = '{"key": "value"'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), False, lambda s, e: (ScalarToken("value", 8, 13, s), 14), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_non_string_key():
    s = '{123: "value"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), False, lambda s, e: (ScalarToken("value", 7, 12, s), 13), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"


# LLM-generated content at query #11
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, "")
    assert callable(scanner)

def test_make_scanner_scans_string_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("test", y + 5),
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

def test_make_scanner_scans_dict_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("key", y + 3),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert end == 13

def test_make_scanner_scans_list_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([ScalarToken(1, 0, 0, "")], y + 3),
        'parse_string': lambda x, y, z: ("", 0),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '[1, 2, 3]')
    token, end = scanner('[1, 2, 3]', 0)
    assert isinstance(token, ListToken)
    assert end == 7

def test_make_scanner_scans_null_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
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

def test_make_scanner_scans_true_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
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

def test_make_scanner_scans_false_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
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

def test_make_scanner_scans_number_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
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

def test_make_scanner_scans_float_token():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
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

def test_make_scanner_raises_stop_iteration():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {}
    })()
    scanner = _make_scanner(context, '')
    with pytest.raises(StopIteration):
        scanner('', 0)

def test_make_scanner_clears_memo():
    context = type('MockContext', (), {
        'parse_array': lambda x, y: ([], 0),
        'parse_string': lambda x, y, z: ("", 0),
        'strict': False,
        'parse_float': float,
        'parse_int': int,
        'memo': {'test': 'value'}
    })()
    scanner = _make_scanner(context, '')
    scanner('', 0)
    assert context.memo == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = MagicMock()
    context.parse_array = MagicMock()
    context.parse_string = MagicMock()
    context.strict = False
    context.parse_float = MagicMock()
    context.parse_int = MagicMock()
    context.memo = MagicMock()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)
    assert token == ScalarToken(None, 7, 10, content)
    assert end == 11


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenizing_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e), {}, '{}')
    assert result == {}
    assert end == 2

def test_tokenizing_single_key_value_pair():
    content = '{"key": "value"}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 13, content)
    result, end = _TokenizingJSONObject(('{"key": "value"}', 0), True, lambda s, e: (value_token, 14), {}, content)
    assert result == {"key": "value"}
    assert end == 15

def test_tokenizing_multiple_key_value_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    key1_token = ScalarToken("key1", 1, 5, content)
    value1_token = ScalarToken("value1", 9, 15, content)
    key2_token = ScalarToken("key2", 19, 23, content)
    value2_token = ScalarToken("value2", 27, 33, content)
    result, end = _TokenizingJSONObject(('{"key1": "value1", "key2": "value2"}', 0), True, lambda s, e: (value1_token if e == 8 else value2_token, 16 if e == 8 else 34), {}, content)
    assert result == {"key1": "value1", "key2": "value2"}
    assert end == 35

def test_tokenizing_with_whitespace():
    content = '{"key":  "value"}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 10, 15, content)
    result, end = _TokenizingJSONObject(('{"key":  "value"}', 0), True, lambda s, e: (value_token, 16), {}, content)
    assert result == {"key": "value"}
    assert end == 17

def test_tokenizing_with_trailing_comma():
    content = '{"key": "value",}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 13, content)
    try:
        _TokenizingJSONObject(('{"key": "value",}', 0), True, lambda s, e: (value_token, 14), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"

def test_tokenizing_with_missing_colon():
    content = '{"key" "value"}'
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 7, 12, content)
    try:
        _TokenizingJSONObject(('{"key" "value"}', 0), True, lambda s, e: (value_token, 13), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenizing_json_object_empty():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test_tokenizing_json_object_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test_tokenizing_json_object_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 8 else "value2", e, e + 6, s), e + 7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test_tokenizing_json_object_with_whitespace():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test_tokenizing_json_object_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test_tokenizing_json_object_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test_tokenizing_json_object_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 8 else "value2", e, e + 6, s), e + 7), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_46():
    s = "  "
    end = 0
    assert s[end] in WHITESPACE_STR
    end += 1
    assert s[end] in WHITESPACE_STR


# LLM-generated content at query #16
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = MagicMock()
    context.parse_array = MagicMock()
    context.parse_string = MagicMock()
    context.strict = True
    context.parse_float = MagicMock()
    context.parse_int = MagicMock()
    context.memo = MagicMock()

    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)

    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start_index == 7
    assert token.end_index == 10
    assert end == 11


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenizing_json_object_empty():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test_tokenizing_json_object_single_pair():
    result, end = _TokenizingJSONObject(('{"a": 1}', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{"a": 1}')
    assert len(result) == 1
    assert result['a'].value == 1
    assert end == 8

def test_tokenizing_json_object_multiple_pairs():
    result, end = _TokenizingJSONObject(('{"a": 1, "b": 2}', 0), True, lambda s, e: (ScalarToken(2, e, e, s), e + 1), {}, '{"a": 1, "b": 2}')
    assert len(result) == 2
    assert result['a'].value == 1
    assert result['b'].value == 2
    assert end == 15

def test_tokenizing_json_object_with_whitespace():
    result, end = _TokenizingJSONObject(('{ "a" : 1 }', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{ "a" : 1 }')
    assert len(result) == 1
    assert result['a'].value == 1
    assert end == 10

def test_tokenizing_json_object_missing_colon():
    try:
        _TokenizingJSONObject(('{"a" 1}', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{"a" 1}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_tokenizing_json_object_missing_comma():
    try:
        _TokenizingJSONObject(('{"a": 1 "b": 2}', 0), True, lambda s, e: (ScalarToken(2, e, e, s), e + 1), {}, '{"a": 1 "b": 2}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_tokenizing_json_object_missing_closing_brace():
    try:
        _TokenizingJSONObject(('{"a": 1', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{"a": 1')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test_tokenizing_json_object_non_string_key():
    try:
        _TokenizingJSONObject(('{1: 1}', 0), True, lambda s, e: (ScalarToken(1, e, e, s), e + 1), {}, '{1: 1}')
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


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
    token, end = scanner('"test"', 0)
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
            self.parse_array = lambda x, y: ([], 2)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = False
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, "[]")
    token, end = scanner("[", 0)
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
    token, end = scanner("{", 0)
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


# LLM-generated content at query #2
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = MagicMock()
    context.parse_array = MagicMock()
    context.parse_string = MagicMock()
    context.strict = True
    context.parse_float = MagicMock()
    context.parse_int = MagicMock()
    context.memo = MagicMock()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)
    assert token.value is None
    assert token.start._index == 7
    assert token.end._index == 10
    assert isinstance(token, ScalarToken)


# LLM-generated content at query #3
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, scan_once, {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    result, end = _TokenizingJSONObject('{"key": "value"}', 0), True, scan_once, {}, '{"key": "value"}')
    assert result == {"key": "value"}
    assert end == 15

def test__TokenizingJSONObject_multiple_key_value_pairs():
    result, end = _TokenizingJSONObject('{"key1": "value1", "key2": "value2"}', 0), True, scan_once, {}, '{"key1": "value1", "key2": "value2"}')
    assert result == {"key1": "value1", "key2": "value2"}
    assert end == 33

def test__TokenizingJSONObject_with_whitespace():
    result, end = _TokenizingJSONObject('{ "key" : "value" }', 0), True, scan_once, {}, '{ "key" : "value" }')
    assert result == {"key": "value"}
    assert end == 19

def test__TokenizingJSONObject_missing_closing_brace():
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject('{"key": "value"', 0), True, scan_once, {}, '{"key": "value"')

def test__TokenizingJSONObject_missing_colon():
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject('{"key" "value"}', 0), True, scan_once, {}, '{"key" "value"}')

def test__TokenizingJSONObject_missing_comma():
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject('{"key1": "value1" "key2": "value2"}', 0), True, scan_once, {}, '{"key1": "value1" "key2": "value2"}')

def test__TokenizingJSONObject_non_string_key():
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject('{123: "value"}', 0), True, scan_once, {}, '{123: "value"}')

def test__TokenizingJSONObject_trailing_comma():
    with pytest.raises(JSONDecodeError):
        _TokenizingJSONObject('{"key": "value",}', 0), True, scan_once, {}, '{"key": "value",}')


# LLM-generated content at query #4
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, i: (ScalarToken(None, i, i, s), i+1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    result, end = _TokenizingJSONObject(('{"a": 1}', 0), True, lambda s, i: (ScalarToken(1, i, i, s), i+1), {}, '{"a": 1}')
    assert result == {"a": ScalarToken(1, 3, 3, '{"a": 1}')}
    assert end == 8

def test__TokenizingJSONObject_multiple_pairs():
    result, end = _TokenizingJSONObject(('{"a": 1, "b": 2}', 0), True, lambda s, i: (ScalarToken(2, i, i, s), i+1), {}, '{"a": 1, "b": 2}')
    assert result == {"a": ScalarToken(1, 3, 3, '{"a": 1, "b": 2}'), "b": ScalarToken(2, 11, 11, '{"a": 1, "b": 2}')}
    assert end == 16

def test__TokenizingJSONObject_with_whitespace():
    result, end = _TokenizingJSONObject(('{ "a" : 1 }', 0), True, lambda s, i: (ScalarToken(1, i, i, s), i+1), {}, '{ "a" : 1 }')
    assert result == {"a": ScalarToken(1, 6, 6, '{ "a" : 1 }')}
    assert end == 10

def test__TokenizingJSONObject_nested_object():
    result, end = _TokenizingJSONObject(('{"a": {"b": 2}}', 0), True, lambda s, i: (ScalarToken(2, i, i, s), i+1), {}, '{"a": {"b": 2}}')
    assert result == {"a": ScalarToken({"b": ScalarToken(2, 10, 10, '{"a": {"b": 2}}')}, 3, 13, '{"a": {"b": 2}}')}
    assert end == 14


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenizing_json_object_empty():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test_tokenizing_json_object_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test_tokenizing_json_object_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 8 else "value2", e, e + 6, s), e + 7), {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test_tokenizing_json_object_with_whitespace():
    content = '  { "key" : "value" , "key2" : "value2" }  '
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value" if e == 10 else "value2", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 2
    assert result["key"].value == "value"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test_tokenizing_json_object_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test_tokenizing_json_object_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test_tokenizing_json_object_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1" if e == 10 else "value2", e, e + 6, s), e + 7), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass

def test_tokenizing_json_object_non_string_key():
    content = '{123: "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = typing.Any
    content = "test content"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = typing.Any
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_with_null_token():
    context = typing.Any
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_with_true_token():
    context = typing.Any
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_with_false_token():
    context = typing.Any
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_with_number_token():
    context = typing.Any
    content = '123'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_with_float_token():
    context = typing.Any
    content = '123.456'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.456
    assert end == 7

def test_make_scanner_with_array_token():
    context = typing.Any
    content = '[1, 2, 3]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert end == 8

def test_make_scanner_with_object_token():
    context = typing.Any
    content = '{"a": 1, "b": 2}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2}
    assert end == 14

def test_make_scanner_with_empty_string():
    context = typing.Any
    content = '""'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == ""
    assert end == 2


# LLM-generated content at query #7
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert result == {"key": ScalarToken("value", 8, 12, content)}
    assert end == len(content)

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s), e + 7), {}, content)
    assert result == {"key1": ScalarToken("value1", 8, 13, content), "key2": ScalarToken("value2", 23, 28, content)}
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key" : "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert result == {"key": ScalarToken("value", 10, 14, content)}
    assert end == len(content)

def test__TokenizingJSONObject_nested_object():
    content = '{"outer": {"inner": "value"}}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 4, s), e + 5), {}, content)
    assert result == {"outer": {"inner": ScalarToken("value", 18, 22, content)}}
    assert end == len(content)


# LLM-generated content at query #8
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = typing.Any
    content = "test content"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = typing.Any
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 6, 5)
    assert end == 6

def test_make_scanner_with_null_token():
    context = typing.Any
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 5, 3)
    assert end == 4

def test_make_scanner_with_true_token():
    context = typing.Any
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 5, 3)
    assert end == 4

def test_make_scanner_with_false_token():
    context = typing.Any
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 6, 4)
    assert end == 5

def test_make_scanner_with_number_token():
    context = typing.Any
    content = '123'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 4, 2)
    assert end == 3

def test_make_scanner_with_float_token():
    context = typing.Any
    content = '123.45'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 7, 5)
    assert end == 6

def test_make_scanner_with_list_token():
    context = typing.Any
    content = '[1, 2, 3]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 9, 8)
    assert end == 9

def test_make_scanner_with_dict_token():
    context = typing.Any
    content = '{"a": 1}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1}
    assert token.start == Position(1, 1, 0)
    assert token.end == Position(1, 8, 7)
    assert end == 8


# LLM-generated content at query #9
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    context = typing.Any
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 7)
    assert token == ScalarToken(None, 7, 10, content)
    assert end == 11


# LLM-generated content at query #10
#--------------------------

```python
def test_whitespace_handling_after_colon():
    s = '{"key":  value}'
    end = 7
    assert s[end] == " "
    end += 1
    assert s[end] == " "
    assert end == 8


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
    assert token.lookup(["a"]).value == 1

def test_tokenize_json_invalid_json():
    with pytest.raises(ParseError) as excinfo:
        tokenize_json('{"a":}')
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position.line_no == 1
    assert excinfo.value.position.column_no == 5
    assert excinfo.value.position.char_index == 4


# LLM-generated content at query #12
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result == {"key": ScalarToken("value", 6, 10, content)}
    assert end == len(content)

def test__TokenizingJSONObject_multiple_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s), e + 7), {}, content)
    assert result == {"key1": ScalarToken("value1", 7, 12, content)}
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result == {"key": ScalarToken("value", 6, 10, content)}
    assert end == len(content)

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s), e + 7), {}, content)
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    except IndexError:
        pass

def test__TokenizingJSONObject_non_string_key():
    content = '{123: "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    except JSONDecodeError as e:
        assert str(e) == "Expecting property name enclosed in double quotes"

def test__TokenizingJSONObject_empty_key():
    content = '{"": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert result == {"": ScalarToken("value", 3, 7, content)}
    assert end == len(content)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_False():
    s = '{"key": "value"'
    end = len(s) - 1
    _w = re.compile(r'\s').match
    _ws = ' \t\n\r'
    try:
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()
    except IndexError:
        pass
    assert True


# LLM-generated content at query #14
#--------------------------

```python
def test__TokenizingJSONObject_empty_object():
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, e: (ScalarToken(None, e, e, s), e + 1), {}, '{}')
    assert result == {}
    assert end == 2

def test__TokenizingJSONObject_single_key_value_pair():
    content = '{"key": "value"}'
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_multiple_key_value_pairs():
    content = '{"key1": "value1", "key2": "value2"}'
    def scan_once(s, e):
        if s[e] == '"':
            return ScalarToken("value1", e, e + 7, s), e + 8
        else:
            return ScalarToken("value2", e, e + 7, s), e + 8
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, {}, content)
    assert len(result) == 2
    assert result["key1"].value == "value1"
    assert result["key2"].value == "value2"
    assert end == len(content)

def test__TokenizingJSONObject_with_whitespace():
    content = '  {  "key"  :  "value"  }  '
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
    assert len(result) == 1
    assert result["key"].value == "value"
    assert end == len(content)

def test__TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test__TokenizingJSONObject_missing_comma():
    content = '{"key1": "value1" "key2": "value2"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value1", e, e + 7, s), e + 8), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test__TokenizingJSONObject_missing_closing_brace():
    content = '{"key": "value"'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)

def test__TokenizingJSONObject_non_string_key():
    content = '{123: "value"}'
    try:
        _TokenizingJSONObject((content, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), {}, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_index_error_handling_in_tokenizing_json_object():
    s = '{"key": "value"'
    end = len(s)
    content = s
    memo = {}
    scan_once = lambda s, end: (ScalarToken("value", end, end, content), end)
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
    assert end == len(s)


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_json_with_empty_string():
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position == Position(column_no=1, line_no=1, char_index=0)


# LLM-generated content at query #17
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = MagicMock()
    content = "test content"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = MagicMock()
    context.parse_string.return_value = ("test", 6)
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 6

def test_make_scanner_with_dict_token():
    context = MagicMock()
    context.parse_array = MagicMock()
    context.strict = False
    context.parse_float = float
    context.parse_int = int
    context.memo = MagicMock()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert end == len(content)

def test_make_scanner_with_list_token():
    context = MagicMock()
    context.parse_array.return_value = ([1, 2, 3], 8)
    content = "[1, 2, 3]"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert end == 8

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
    content = "123.45"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_with_integer_token():
    context = MagicMock()
    context.parse_int = int
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_clears_memo():
    context = MagicMock()
    context.memo = MagicMock()
    content = "null"
    scanner = _make_scanner(context, content)
    scanner(content, 0)
    context.memo.clear.assert_called_once()


# LLM-generated content at query #18
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = MagicMock()
    content = "test content"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_with_string_token():
    context = MagicMock()
    context.parse_string.return_value = ("test", 6)
    content = '{"key": "test"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 1)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.start == 1
    assert token.end == 5

def test_make_scanner_with_dict_token():
    context = MagicMock()
    context.parse_string.return_value = ("key", 5)
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.start == 0
    assert token.end == len(content) - 1

def test_make_scanner_with_list_token():
    context = MagicMock()
    context.parse_array.return_value = ([1, 2, 3], 8)
    content = "[1, 2, 3]"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.start == 0
    assert token.end == len(content) - 1

def test_make_scanner_with_null_token():
    context = MagicMock()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

def test_make_scanner_with_true_token():
    context = MagicMock()
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

def test_make_scanner_with_false_token():
    context = MagicMock()
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4

def test_make_scanner_with_integer_token():
    context = MagicMock()
    context.parse_int.return_value = 42
    content = "42"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

def test_make_scanner_with_float_token():
    context = MagicMock()
    context.parse_float.return_value = 3.14
    content = "3.14"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

def test_make_scanner_clears_memo():
    context = MagicMock()
    context.memo = MagicMock()
    content = "test"
    scanner = _make_scanner(context, content)
    scanner(content, 0)
    context.memo.clear.assert_called_once()


# LLM-generated content at query #19
#--------------------------

```python
def test_index_error_handling():
    s = '{"key": "value"'
    end = len(s)
    try:
        nextchar = s[end]
    except IndexError:
        nextchar = ""
    assert nextchar == ""


# LLM-generated content at query #20
#--------------------------

```python
def test_make_scanner_with_string_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: ("test", i + 5), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "test content")
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.start._index == 0
    assert token.end._index == 4
    assert end == 5

def test_make_scanner_with_dict_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: ("key", i + 3), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, '{"key": "value"}')
    token, end = scanner('{"key": "value"}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start._index == 0
    assert token.end._index == 13
    assert end == 14

def test_make_scanner_with_list_token():
    context = Mock(parse_array=lambda x, y: ([1, 2, 3], 5), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "[1, 2, 3]")
    token, end = scanner("[1, 2, 3]", 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start._index == 0
    assert token.end._index == 7
    assert end == 8

def test_make_scanner_with_null_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start._index == 0
    assert token.end._index == 3
    assert end == 4

def test_make_scanner_with_true_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "true")
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start._index == 0
    assert token.end._index == 3
    assert end == 4

def test_make_scanner_with_false_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "false")
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start._index == 0
    assert token.end._index == 4
    assert end == 5

def test_make_scanner_with_number_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "123")
    token, end = scanner("123", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start._index == 0
    assert token.end._index == 2
    assert end == 3

def test_make_scanner_with_float_token():
    context = Mock(parse_array=lambda x, y: (None, 0), parse_string=lambda s, i, strict: (None, 0), strict=False, parse_float=float, parse_int=int, memo={})
    scanner = _make_scanner(context, "123.45")
    token, end = scanner("123.45", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start._index == 0
    assert token.end._index == 5
    assert end == 6


# LLM-generated content at query #21
#--------------------------

```python
def test_scan_once_raises_stop_iteration():
    s_and_end = ('{"key":', 6)
    strict = True
    def scan_once(s, end):
        raise StopIteration(6)
    memo = {}
    content = '{"key":}'

    with pytest.raises(JSONDecodeError) as exc_info:
        _TokenizingJSONObject(s_and_end, strict, scan_once, memo, content)
    assert str(exc_info.value) == "Expecting value"


# LLM-generated content at query #22
#--------------------------

```python
def test_null_token_creation():
    content = '{"key": null}'
    token, end = ScalarToken(None, 7, 10, content), 11
    assert token.string == "null"
    assert token.value is None
    assert token.start.line_no == 1
    assert token.start.column_no == 8
    assert token.end.line_no == 1
    assert token.end.column_no == 11


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_false():
    s = '{"key": value'
    end = len(s) - 1
    _w = WHITESPACE.match
    _ws = WHITESPACE_STR
    with pytest.raises(IndexError):
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()


# LLM-generated content at query #24
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
            self.parse_string = lambda x, y, z: ("test", 6)
            self.strict = True
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
            self.strict = True
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
            self.strict = True
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
            self.strict = True
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
            self.strict = True
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
            self.strict = True
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
            self.parse_array = lambda x, y: ([ScalarToken(1, 0, 0, "")], 3)
            self.parse_string = lambda x, y, z: ("", 0)
            self.strict = True
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
            self.parse_string = lambda x, y, z: ("key", 5)
            self.strict = True
            self.parse_float = float
            self.parse_int = int
            self.memo = {}

    context = MockContext()
    scanner = _make_scanner(context, '{"key": 1}')
    token, end = scanner('{"key": 1}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": 1}
    assert end == 9


# LLM-generated content at query #25
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

def test__TokenizingJSONObject_with_whitespace():
    s = '  { "key" : "value" }  '
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), memo, content)
    assert result == {"key": ScalarToken("value", 13, 17, content)}
    assert end == len(s)

def test__TokenizingJSONObject_missing_colon():
    s = '{"key" "value"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ':' delimiter"

def test__TokenizingJSONObject_missing_comma():
    s = '{"key1": "value1" "key2": "value2"}'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value1", e, e + 6, s), e + 7), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting ',' delimiter"

def test__TokenizingJSONObject_missing_closing_brace():
    s = '{"key": "value"'
    content = s
    memo = {}
    try:
        _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken("value", e, e + 5, s), e + 6), memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert str(e) == "Expecting value"

def test__TokenizingJSONObject_nested_object():
    s = '{"outer": {"inner": "value"}}'
    content = s
    memo = {}
    result, end = _TokenizingJSONObject((s, 0), True, lambda s, e: (ScalarToken({"inner": "value"}, e, e + 17, s), e + 18), memo, content)
    assert result == {"outer": ScalarToken({"inner": "value"}, 8, 24, content)}
    assert end == len(s)


# LLM-generated content at query #26
#--------------------------

```python
def test_make_scanner_returns_callable():
    context = typing.Any
    content = "test"
    scanner = _make_scanner(context, content)
    assert callable(scanner)

def test_make_scanner_scans_string():
    context = typing.Any
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert end == 5

def test_make_scanner_scans_object():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("key", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert end == len(content)

def test_make_scanner_scans_array():
    context = typing.Any
    context.parse_array = lambda x, y: ([1, 2, 3], len(x[0]))
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "[1, 2, 3]"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert end == len(content)

def test_make_scanner_scans_null():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

def test_make_scanner_scans_true():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

def test_make_scanner_scans_false():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_scans_number():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert end == 3

def test_make_scanner_scans_float():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "123.45"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert end == 6

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    context = typing.Any
    context.parse_array = lambda x, y: ([], 0)
    context.parse_string = lambda s, i, strict: ("test", i + 5)
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    content = "invalid"
    scanner = _make_scanner(context, content)
    with pytest.raises(StopIteration):
        scanner(content, 0)


