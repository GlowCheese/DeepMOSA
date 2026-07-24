####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_make_scanner_with_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert idx == 4

def test_make_scanner_with_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "true"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert idx == 4

def test_make_scanner_with_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "false"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert idx == 5

def test_make_scanner_with_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "123"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert idx == 3

def test_make_scanner_with_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "12.34"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34
    assert token.string == "12.34"
    assert idx == 5

def test_make_scanner_with_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("parsed", idx + 8)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = '"string"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "parsed"
    assert token.string == '"string"'
    assert idx == 8

def test_make_scanner_with_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "{}"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert idx == 2

def test_make_scanner_with_empty_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "[]"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert idx == 2

def test_make_scanner_clears_memo():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {"key": "value"}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan_once: ([], args[1])
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #2
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    expected = {}
    assert result == expected
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 9, 15, content)
        return token, 16
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 9, 15, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 17

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, 7, 7, content)
            return token, 8
        else:
            token = ScalarToken(2, 15, 15, content)
            return token, 16
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 7, 7, content)
    key_token_b = ScalarToken("b", 11, 12, content)
    value_token_2 = ScalarToken(2, 15, 15, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == 17

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 13, 19, content)
        return token, 20
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 3, 6, content)
    value_token = ScalarToken("value", 13, 19, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 21

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 9, 15, content), 16
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ':' delimiter"

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, 7, 7, content)
            return token, 8
        else:
            token = ScalarToken(2, 13, 13, content)
            return token, 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_quote():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 7, 13, content), 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_stop_iteration():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #3
#--------------------------

def test_tokenize_json_empty_string():
    content = ""
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    content = b""
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_null():
    content = "null"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_true():
    content = "true"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_false():
    content = "false"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_number_integer():
    content = "42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_number_float():
    content = "3.14"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_string():
    content = '"hello"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_array():
    content = '[1, "two", false]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", False]
    assert token.string == '[1, "two", false]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_object():
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested_structure():
    content = '{"list": [1, 2], "nested": {"inner": true}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    expected = {"list": [1, 2], "nested": {"inner": True}}
    assert token.value == expected
    assert token.string == '{"list": [1, 2], "nested": {"inner": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=44, char_index=43)

def test_tokenize_json_bytes_input():
    content = b'{"test": 123}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"test": 123}
    assert token.string == '{"test": 123}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=13, char_index=12)

def test_tokenize_json_parse_error():
    content = '{"invalid": json}'
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index > 0

def test_tokenize_json_whitespace_only():
    content = "   \n\t  "
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_multiline():
    content = '[\n  1,\n  2\n]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)


# LLM-generated content at query #4
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_scalar_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_scalar_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import re
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_scalar_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import re
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '{}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_empty_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_array = lambda self, idx_scan: ([], idx_scan[1])
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '[]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_nested_structure():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken, ListToken, ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("value", idx + 7)
        parse_array = lambda self, idx_scan: ([ScalarToken("item", idx_scan[1]+1, idx_scan[1]+4, content)], idx_scan[1]+6)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '{"key": ["item"]}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert "key" in token.value
    assert isinstance(token.value["key"], list)
    assert token.value["key"][0] == "item"
    assert end == len(content)


# LLM-generated content at query #5
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
    memo = {}
    result, end = _TokenizingJSONObject(('{}', 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken('value', start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 1, 5, content)
    value_token = ScalarToken('value', 9, 15, content)
    assert list(result.keys())[0] == key_token
    assert result[key_token] == value_token
    assert end == len(content)

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken('a', 1, 3, content)
    key_token_b = ScalarToken('b', 8, 10, content)
    assert result[key_token_a] == ScalarToken(1, 7, 7, content)
    assert result[key_token_b] == ScalarToken(2, 14, 14, content)
    assert end == len(content)

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken('value', start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 2, 6, content)
    value_token = ScalarToken('value', 11, 17, content)
    assert list(result.keys())[0] == key_token
    assert result[key_token] == value_token
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken('value', idx, idx + 5, content), idx + 6
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ':' delimiter"

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken('value', idx, idx + 5, content), idx + 6
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value", "key": "value2"}'
    memo = {}
    call_count = [0]
    def scan_once(s, idx):
        call_count[0] += 1
        if call_count[0] == 1:
            return ScalarToken('value', idx, idx + 5, content), idx + 6
        else:
            return ScalarToken('value2', idx, idx + 6, content), idx + 7
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 1, 5, content)
    assert len(result) == 1
    assert result[key_token] == ScalarToken('value2', 23, 28, content)
    assert end == len(content)


# LLM-generated content at query #6
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    try:
        tokenize_json(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    try:
        tokenize_json("   ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

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

def test_tokenize_json_empty_object():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_object():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_empty_array():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_array():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"a": [{"b": true}]}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [{"b": True}]}
    assert token.string == '{"a": [{"b": true}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=21, char_index=20)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"test": 123}')
    assert isinstance(token, DictToken)
    assert token.value == {"test": 123}
    assert token.string == '{"test": 123}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=13, char_index=12)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_invalid_json_unexpected_token():
    try:
        tokenize_json('{"key": }')
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.line_no == 1

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)


# LLM-generated content at query #7
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject((content, 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 5, content)
    value_token = ScalarToken("value", 9, 15, content)
    assert result == {key_token: value_token}
    assert end == 17

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 3, content)
    key_token_b = ScalarToken("b", 8, 10, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 15

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 6, content)
    value_token = ScalarToken("value", 11, 17, content)
    assert result == {key_token: value_token}
    assert end == 19

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_invalid_key():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_missing_value():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_json_does_not_raise_parse_error_on_valid_json():
    content = '{"key": "value"}'
    decoder = _TokenizingDecoder(content=content)
    result = decoder.decode(content)
    assert isinstance(result, Token)


# LLM-generated content at query #9
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        memo = {}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_scalar_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_scalar_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_int = int
        parse_float = float
        memo = {}
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_scalar_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_int = int
        parse_float = float
        memo = {}
    context = MockContext()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    def mock_parse_object(string_idx, strict, scan_once, memo, content):
        return {}, 2
    MockContext.parse_object = staticmethod(mock_parse_object)
    context = MockContext()
    content = '{}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = lambda self, string_idx, scan_once: ([], 2)
        memo = {}
    context = MockContext()
    content = '[]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_stop_iteration_on_empty_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = ''
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_memo_cleared_after_scan():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        memo = {'key': 'value'}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #10
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_scalar_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_scalar_number_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        parse_float = float
        parse_int = int
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_scalar_number_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        parse_float = float
        parse_int = int
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            return "key", idx + 5
        def parse_array(self, args, scan_once):
            pass
        parse_object = _TokenizingJSONObject
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert end == len(content)

def test_make_scanner_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            return "item", idx + 6
        def parse_array(self, args, scan_once):
            return [ScalarToken("item", 1, 6, content)], 8
    context = MockContext()
    content = '["item"]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == ["item"]
    assert token.string == '["item"]'
    assert end == len(content)

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {"key": "value"}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration_on_empty_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = ''
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_stop_iteration_on_invalid_char():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        memo = {}
        parse_float = float
        parse_int = int
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'invalid'
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #11
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_simple_key_value():
    content = '{"key": 123}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(123, idx, idx + 2, content)
        return token, idx + 3
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken(123, 7, 9, content)
    assert result == {key_token: value_token}
    assert end == 11

def test_TokenizingJSONObject_with_whitespace():
    content = ' { "key" : 123 } '
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(123, idx, idx + 2, content)
        return token, idx + 3
    result, end = _TokenizingJSONObject((content, 1), True, scan_once, memo, content)
    key_token = ScalarToken("key", 3, 6, content)
    value_token = ScalarToken(123, 10, 12, content)
    assert result == {key_token: value_token}
    assert end == 15

def test_TokenizingJSONObject_multiple_pairs():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        else:
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 5, 5, content)
    key_token_b = ScalarToken("b", 9, 10, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 15

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" 123}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx + 2, content), idx + 3
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ':' delimiter"

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        else:
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_quote():
    content = '{key: 123}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx + 2, content), idx + 3
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_stop_iteration():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken(s[start+1:idx-1], start, idx-1, content)
            return token, idx
        else:
            raise StopIteration(idx)
    def scanstring(s, idx, strict):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            return s[start+1:idx-1], idx
        else:
            raise json.JSONDecodeError("Expecting property name", s, idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"


# LLM-generated content at query #13
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    try:
        tokenize_json("   ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

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

def test_tokenize_json_empty_object():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_object():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_empty_array():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_array():
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == '[1, 2, 3]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"a": [{"b": true}]}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [{"b": True}]}
    assert token.string == '{"a": [{"b": true}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=21, char_index=20)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'"bytes"')
    assert isinstance(token, ScalarToken)
    assert token.value == "bytes"
    assert token.string == '"bytes"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json('{"invalid": }')
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting value" in e.text

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)


# LLM-generated content at query #14
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        memo = {}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_returns_dict_token_for_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    def mock_parse_object(*args):
        return ({"key": ScalarToken("value", 8, 12, '{"key": "value"}')}, 15)
    MockContext.parse_object = mock_parse_object
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert end == 15

def test_make_scanner_returns_list_token_for_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = lambda self, string_idx, scan_once: ([ScalarToken(1, 2, 2, '[1]')], 3)
        memo = {}
    context = MockContext()
    content = '[1]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1]
    assert token.string == '[1]'
    assert end == 3

def test_make_scanner_returns_scalar_token_for_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_returns_scalar_token_for_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import re
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    MockContext.match_number = re.compile(r'(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)').match
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_clears_memo_after_scan():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        memo = {"key": "value"}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #15
#--------------------------

def test_make_scanner_parse_object_is_not_TokenizingJSONObject():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    from unittest.mock import Mock
    context = Mock()
    context.parse_array = Mock()
    context.parse_string = Mock()
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    scanner = _make_scanner(context, "")
    assert scanner.__closure__[0].cell_contents != _TokenizingJSONObject


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken(s[start+1:idx-1], start, idx-1, content)
            return token, idx
        else:
            raise StopIteration(idx)
    def whitespace_match(s, idx):
        class Match:
            def end(self):
                return idx
        return Match()
    result = _TokenizingJSONObject((content, 0), True, scan_once, memo, content, _w=whitespace_match, _ws='')
    assert result[0] == {'key': ScalarToken('value', 8, 14, content)}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    expected = {}
    assert result == expected
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": 123}'
    memo = {}
    def scan_once(s, idx):
        if s[idx:].startswith("123"):
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken(123, 8, 10, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 12

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        if s[idx:].startswith("1"):
            token = ScalarToken(1, idx, idx, content)
            call_count += 1
            return token, idx+1
        if s[idx:].startswith("2"):
            token = ScalarToken(2, idx, idx, content)
            call_count += 1
            return token, idx+1
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    key_token_b = ScalarToken("b", 8, 9, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == 16
    assert call_count == 2

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    memo = {}
    def scan_once(s, idx):
        if s[idx:].startswith("123"):
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken(123, 10, 12, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 15

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" 123}'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        if s[idx:].startswith("1"):
            token = ScalarToken(1, idx, idx, content)
            call_count += 1
            return token, idx+1
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: 123}'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_stop_iteration_value():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from unittest.mock import Mock, patch

    mock_scan_once = Mock(side_effect=[(ScalarToken("value", 10, 14, '{"key":"value"}'), 15)])
    memo = {}
    content = '{"key":"value"}'
    result = _TokenizingJSONObject(('{"key":"value"}', 0), True, mock_scan_once, memo, content)
    assert result == ({"key": ScalarToken("value", 10, 14, content)}, 16)


# LLM-generated content at query #3
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert idx == 6

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert idx == 4

def test_make_scanner_scalar_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert idx == 5

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert idx == 4

def test_make_scanner_scalar_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert idx == 2

def test_make_scanner_scalar_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert idx == 4

def test_make_scanner_dict_token():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "key", idx + 5
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '{"key": "value"}'
    def mock_parse_object(args, strict, scan_once, memo, content):
        return {ScalarToken("key", 1, 5, content): ScalarToken("value", 8, 14, content)}, 16
    _TokenizingJSONObject = mock_parse_object
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert idx == 16

def test_make_scanner_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken, ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            return [ScalarToken(1, 1, 1, content), ScalarToken(2, 3, 3, content)], 5
    context = MockContext()
    content = '[1, 2]'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[1, 2]'
    assert idx == 5

def test_make_scanner_stop_iteration_on_empty_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            pass
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = ''
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_memo_cleared_after_scan():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {'test': 1}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #4
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert idx == 6

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert idx == 4

def test_make_scanner_scalar_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert idx == 5

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert idx == 4

def test_make_scanner_scalar_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '42'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert idx == 2

def test_make_scanner_scalar_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert idx == 4

def test_make_scanner_list():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = lambda self, string_idx, scan_once: ([ScalarToken(1, 1, 1, ''), ScalarToken(2, 3, 3, '')], 5)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '[1, 2]'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[1, 2]'
    assert idx == 5

def test_make_scanner_dict():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken, ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    def mock_parse_object(string_idx, strict, scan_once, memo, content):
        key = ScalarToken("key", 1, 3, content)
        value = ScalarToken("value", 5, 9, content)
        return {key: value}, 11
    context = MockContext()
    context.parse_object = mock_parse_object
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert idx == 11

def test_make_scanner_empty_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = ''
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        parse_float = float
        parse_int = int
        memo = {'key': 'value'}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #5
#--------------------------

def test_tokenize_json_empty_string():
    content = ""
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    content = b""
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_string():
    content = "   \n\t  "
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_scalar_null():
    content = "null"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_true():
    content = "true"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_false():
    content = "false"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_scalar_integer():
    content = "42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_scalar_float():
    content = "3.14"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_string():
    content = '"hello"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list_empty():
    content = "[]"
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_list_simple():
    content = '[1, "two", false]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", False]
    assert token.string == '[1, "two", false]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_dict_empty():
    content = "{}"
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_dict_simple():
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested_structure():
    content = '{"list": [1, 2], "nested": {"inner": true}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    expected = {"list": [1, 2], "nested": {"inner": True}}
    assert token.value == expected
    assert token.string == '{"list": [1, 2], "nested": {"inner": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=44, char_index=43)

def test_tokenize_json_bytes_input():
    content = b'{"test": 123}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"test": 123}
    assert token.string == '{"test": 123}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=13, char_index=12)

def test_tokenize_json_invalid_json():
    content = '{"unclosed": '
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index == 13

def test_tokenize_json_invalid_bytes():
    content = b'{"invalid": \x80}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"invalid": ""}

def test_tokenize_json_multiline():
    content = '[\n    1,\n    2\n]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n    1,\n    2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=13)

def test_tokenize_json_position_calculation():
    content = '{\n  "a": 1\n}'
    token = tokenize_json(content)
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=1, char_index=9)
    child = token.lookup(["a"])
    assert child.start == Position(line_no=2, column_no=6, char_index=7)
    assert child.end == Position(line_no=2, column_no=6, char_index=7)


# LLM-generated content at query #6
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json._make_scanner.__closure__[0].cell_contents
    parse_object = context.parse_object
    _TokenizingJSONObject = typesystem.tokenize.tokenize_json._TokenizingJSONObject
    result = parse_object is _TokenizingJSONObject
    assert result is False


# LLM-generated content at query #7
#--------------------------

def test_make_scanner_parse_object_not_tokenizing_json_object():
    context = type('Context', (), {'parse_array': lambda x, y: ([], 0), 'parse_string': lambda x, y, z: ('', 0), 'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = ''
    scanner = _make_scanner(context, content)
    result = scanner('{}', 0)
    assert not isinstance(result[0], DictToken)


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken(s[start+1:idx-1], start, idx-1, content)
            return token, idx
        else:
            raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    assert end == len(content)


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_61_evaluates_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            token, end = json.decoder.scanstring(s, idx + 1, json.decoder.strict)
            return ScalarToken(token, idx, end - 1, content), end
        raise StopIteration(idx)
    s = content
    end = 0
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"].value == "value"
    assert new_end == len(content)


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_48_evaluates_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    import typesystem.tokenize.tokenize_json as tokenize_json_module
    original_scanstring = tokenize_json_module.scanstring
    mock_scanstring = lambda s, end, strict: ("key", end + 5)
    tokenize_json_module.scanstring = mock_scanstring
    memo = {}
    content = '{"key": "value"}'
    s = content
    end = 1
    def mock_scan_once(s, end):
        token = ScalarToken("value", end, end + 5, content)
        return token, end + 6
    try:
        result, new_end = _TokenizingJSONObject((s, end), True, mock_scan_once, memo, content)
        assert result == {"key": "value"}
    finally:
        tokenize_json_module.scanstring = original_scanstring


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_48_evaluates_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    import typesystem.tokenize.tokenize_json as module
    original_scan_once = module.scan_once
    mock_scan_once = lambda s, idx: (ScalarToken("value", idx, idx + 4, s), idx + 5)
    module.scan_once = mock_scan_once
    content = '{"key": "value"}'
    memo = {}
    result = _TokenizingJSONObject((content, 0), True, mock_scan_once, memo, content)
    module.scan_once = original_scan_once
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert result[0] == {"key": "value"}
    assert result[1] == len(content)


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_61_evaluates_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    import typing
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            value = s[start+1:idx-1]
            token = ScalarToken(value, start, idx-1, content)
            return token, idx
        else:
            raise StopIteration(idx)
    def whitespace_match(s, idx):
        class Match:
            def end(self):
                return idx
        return Match()
    result, end_index = _TokenizingJSONObject((content, 0), True, scan_once, memo, content, _w=whitespace_match, _ws=' \t\n\r')
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.string == '"key"'
    assert key_token.value == 'key'
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.string == '"value"'
    assert value_token.value == 'value'


# LLM-generated content at query #13
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject(("{}", 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    value_token = list(result.values())[0]
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 2
    keys = [k.string for k in result.keys()]
    assert '"a"' in keys
    assert '"b"' in keys
    values = [v.value for v in result.values()]
    assert 1 in values
    assert 2 in values
    assert end == len(content)

def test_TokenizingJSONObject_whitespace_handling():
    content = '{  "key"  :  "value"  }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    value_token = list(result.values())[0]
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value", "key": "value2"}'
    memo = {}
    call_count = [0]
    def scan_once(s, idx):
        call_count[0] += 1
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        value = "value" if call_count[0] == 1 else "value2"
        return ScalarToken(value, start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    value_token = list(result.values())[0]
    assert value_token.string == '"value2"'
    assert value_token.value == "value2"
    assert end == len(content)


# LLM-generated content at query #14
#--------------------------

def test_make_scanner_with_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert idx == 4

def test_make_scanner_with_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "true"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert idx == 4

def test_make_scanner_with_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "false"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert idx == 5

def test_make_scanner_with_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "42"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert idx == 2

def test_make_scanner_with_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "3.14"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert idx == 4

def test_make_scanner_with_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: (string[idx:idx+5], idx+6)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '"hello"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert idx == 7

def test_make_scanner_with_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_object = lambda self, args, strict, scan_once, memo, content: ({}, args[1])
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "{}"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert idx == 2

def test_make_scanner_with_empty_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_array = lambda self, args, scan_once: ([], args[1])
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "[]"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert idx == 2

def test_make_scanner_with_nested_structure():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken, ListToken, ScalarToken
    class MockContext:
        strict = True
        parse_object = lambda self, args, strict, scan_once, memo, content: ({"key": ScalarToken("value", 0, 0, content)}, args[1]+10)
        parse_array = lambda self, args, scan_once: ([ScalarToken(1, 0, 0, content)], args[1]+5)
        parse_string = lambda self, string, idx, strict: (string[idx:idx+5], idx+6)
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert idx == 16
    content = '[1]'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert idx == 3

def test_make_scanner_stop_iteration_on_invalid():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = "invalid"
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration:
        pass


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_48_evaluates_false():
    s = '{"key": "value"}'
    end = len('{"key":')
    _ws = " \t\n\r"
    result = s[end] in _ws
    assert result == False


# LLM-generated content at query #16
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    try:
        tokenize_json(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_string():
    try:
        tokenize_json("   ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_bytes():
    try:
        tokenize_json(b"   ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

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

def test_tokenize_json_empty_list():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_list_with_elements():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_empty_dict():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_dict_with_elements():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"list": [1, 2], "nested": {"inner": true}}')
    assert isinstance(token, DictToken)
    expected = {"list": [1, 2], "nested": {"inner": True}}
    assert token.value == expected
    assert token.string == '{"list": [1, 2], "nested": {"inner": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=46, char_index=45)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"test": 123}')
    assert isinstance(token, DictToken)
    assert token.value == {"test": 123}
    assert token.string == '{"test": 123}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=13, char_index=12)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting property name enclosed in double quotes" in e.text
        assert isinstance(e.position, Position)

def test_tokenize_json_incomplete_json():
    try:
        tokenize_json('{"key":')
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)

def test_tokenize_json_multiline():
    json_str = '{\n  "name": "John",\n  "age": 30\n}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.string == json_str
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=30)

def test_tokenize_json_with_unicode():
    token = tokenize_json('"café"')
    assert isinstance(token, ScalarToken)
    assert token.value == "café"
    assert token.string == '"café"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_bytes_with_unicode():
    token = tokenize_json(b'"caf\xc3\xa9"')
    assert isinstance(token, ScalarToken)
    assert token.value == "café"
    assert token.string == '"café"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)


# LLM-generated content at query #17
#--------------------------

def test_tokenize_json_scalar_string():
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_scalar_number():
    token = tokenize_json('42')
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_scalar_float():
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_true():
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_false():
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_scalar_null():
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_list():
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == '[1, 2, 3]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_dict():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested():
    token = tokenize_json('{"a": [1, {"b": 2}]}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, {"b": 2}]}
    assert token.string == '{"a": [1, {"b": 2}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=21, char_index=20)

def test_tokenize_json_bytes():
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_empty_string():
    try:
        tokenize_json('')
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    try:
        tokenize_json('   ')
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json('{invalid}')
    except ParseError as e:
        assert e.code == "parse_error"
        assert isinstance(e.position, Position)

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)

def test_tokenize_json_lookup():
    token = tokenize_json('{"a": [1, 2]}')
    child = token.lookup(["a", 1])
    assert isinstance(child, ScalarToken)
    assert child.value == 2
    assert child.string == '2'
    assert child.start == Position(line_no=1, column_no=12, char_index=11)
    assert child.end == Position(line_no=1, column_no=12, char_index=11)

def test_tokenize_json_lookup_key():
    token = tokenize_json('{"key": "value"}')
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert key_token.string == '"key"'
    assert key_token.start == Position(line_no=1, column_no=2, char_index=1)
    assert key_token.end == Position(line_no=1, column_no=6, char_index=5)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from unittest.mock import Mock, patch
    import json

    content = '{"key": "value"}'
    s = content
    end = 0
    strict = True
    memo = {}
    scan_once = Mock(return_value=(ScalarToken("value", 8, 14, content), 15))
    with patch('typesystem.tokenize.tokenize_json.scanstring', return_value=("key", 6)):
        with patch('typesystem.tokenize.tokenize_json.WHITESPACE.match') as mock_w:
            mock_w.return_value.end.return_value = 0
            result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
    assert result == {"key": ScalarToken("value", 8, 14, content)}
    assert new_end == 16


# LLM-generated content at query #19
#--------------------------

def test_tokenize_json_empty_string_raises_parse_error():
    content = ""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_scanner_parse_object_is_not_TokenizingJSONObject():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from unittest.mock import Mock
    context = Mock()
    context.parse_array = Mock()
    context.parse_string = Mock()
    context.strict = True
    context.parse_float = float
    context.parse_int = int
    context.memo = {}
    scanner = _make_scanner(context, "")
    assert scanner.__closure__ is not None
    cell_values = [cell.cell_contents for cell in scanner.__closure__]
    parse_object_in_closure = None
    for val in cell_values:
        if callable(val) and val.__name__ == '_scan_once':
            closure_of_inner = val.__closure__
            if closure_of_inner:
                inner_cell_values = [cell.cell_contents for cell in closure_of_inner]
                for inner_val in inner_cell_values:
                    if inner_val is _TokenizingJSONObject:
                        parse_object_in_closure = inner_val
    assert parse_object_in_closure is not _TokenizingJSONObject


# LLM-generated content at query #21
#--------------------------

def test_parse_object_not_equal_to_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    parse_object = typesystem.tokenize.tokenize_json._make_scanner(None, "").__closure__[0].cell_contents
    _TokenizingJSONObject = typesystem.tokenize.tokenize_json._TokenizingJSONObject
    assert parse_object is not _TokenizingJSONObject


# LLM-generated content at query #22
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject(("{}", 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = list(result.keys())[0]
    value_token = result[key_token]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    keys = list(result.keys())
    values = list(result.values())
    assert keys[0].string == '"a"'
    assert keys[0].value == "a"
    assert values[0].value == 1
    assert keys[1].string == '"b"'
    assert keys[1].value == "b"
    assert values[1].value == 2
    assert end == len(content)

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        return ScalarToken("value", start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = list(result.keys())[0]
    value_token = result[key_token]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_missing_value():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_json_empty_string_raises_parse_error():
    content = ""
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken(s[start+1:idx-1], start, idx-1, content)
            return token, idx
        else:
            raise StopIteration(idx)
    s_and_end = (content, 1)
    result, end = _TokenizingJSONObject(s_and_end, True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.string == 'key'
    assert key_token.value == 'key'
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.string == 'value'
    assert value_token.value == 'value'
    assert end == len(content)


# LLM-generated content at query #25
#--------------------------

def test_tokenize_json_empty_string_raises_parse_error():
    content = ""
    try:
        tokenize_json(content)
        assert False
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)


# LLM-generated content at query #26
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    s = '{"key": "value"}'
    end = 7
    _ws = " \t\n\r"
    result = s[end] in _ws
    assert result == False


# LLM-generated content at query #27
#--------------------------

def test_make_scanner_parse_object_not_equal_to_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json._make_scanner.__closure__[0].cell_contents
    parse_object = context.parse_object
    _TokenizingJSONObject = typesystem.tokenize.tokenize_json._TokenizingJSONObject
    result = parse_object is _TokenizingJSONObject
    assert not result


# LLM-generated content at query #28
#--------------------------

def test_tokenizing_json_object_predicate_at_line_61_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken(s[start+1:idx-1], start, idx-1, content)
            return token, idx
        raise StopIteration(idx)
    def scanstring(s, idx, strict):
        start = idx
        idx += 1
        while s[idx] != '"':
            idx += 1
        idx += 1
        return s[start+1:idx-1], idx
    import re
    WHITESPACE = re.compile(r'\s*')
    WHITESPACE_STR = ' \t\n\r'
    def _w(s, idx):
        return WHITESPACE.match(s, idx)
    s = content
    end = 0
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content, _w, WHITESPACE_STR)
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    assert new_end == len(content)


# LLM-generated content at query #29
#--------------------------

def test_parse_object_not_used_when_not_dict():
    context = type('Context', (), {'parse_array': lambda x, y: ([], 0), 'parse_string': lambda x, y, z: ('', 0), 'strict': False, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, '')
    token, end = scanner('[]', 0)
    assert isinstance(token, ListToken)


