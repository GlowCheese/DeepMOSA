####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_object = None
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

def test_make_scanner_returns_scalar_token_for_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_object = None
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
        parse_object = None
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
        parse_object = None
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
        parse_object = None
        parse_array = None
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

def test_make_scanner_returns_dict_token_for_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken, ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("key", idx + 5)
        parse_object = lambda self, args, strict, scan_once, memo, content: ({ScalarToken("key", 1, 5, content): ScalarToken("value", 7, 13, content)}, 14)
        parse_array = None
        memo = {}
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
    from typesystem.tokenize.tokens import ListToken, ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_object = None
        parse_array = lambda self, args, scan_once: ([ScalarToken(1, 1, 1, content), ScalarToken(2, 3, 3, content)], 5)
        memo = {}
    context = MockContext()
    content = '[1, 2]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[1, 2]'
    assert end == 6

def test_make_scanner_clears_memo_after_scan():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_object = None
        parse_array = None
        memo = {"key": "value"}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = None
        parse_object = None
        parse_array = None
        memo = {}
    context = MockContext()
    content = 'invalid'
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #2
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
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 14, content)
    assert result == {key_token: value_token}
    assert end == 16

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            return ScalarToken(1, idx, idx, content), idx + 1
        else:
            return ScalarToken(2, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    key_token_b = ScalarToken("b", 8, 9, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 15

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken("value", 10, 16, content)
    assert result == {key_token: value_token}
    assert end == 18

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 5, content), idx + 6
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
        return ScalarToken("value", idx, idx + 5, content), idx + 6
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


# LLM-generated content at query #3
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject(("{}", 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    expected = {}
    assert result == expected
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 5, content)
    value_token = ScalarToken("value", 9, 15, content)
    expected = {key_token: value_token}
    assert result == expected
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
    value_token_1 = ScalarToken(1, 7, 7, content)
    key_token_b = ScalarToken("b", 10, 12, content)
    value_token_2 = ScalarToken(2, 16, 16, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == 18

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 6, content)
    value_token = ScalarToken("value", 12, 18, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 20

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 5, content), idx + 6
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
        return ScalarToken("value", idx, idx + 5, content), idx + 6
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


# LLM-generated content at query #4
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_simple_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value", start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.value == "key"
    value_token = result[key_token]
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_multiple_pairs():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        elif s[idx] == '2':
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 2
    assert call_count == 2
    values = [v.value for v in result.values()]
    assert set(values) == {1, 2}
    assert end == len(content)

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value", start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.value == "key"
    value_token = result[key_token]
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, 0, 0, content), idx
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
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        elif s[idx] == '2':
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, 0, 0, content), idx
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value", "key": "value2"}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value" if call_count == 1 else "value2", start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 2
    keys = list(result.keys())
    assert keys[0].value == "key"
    assert keys[1].value == "key"
    assert keys[0] is keys[1]
    assert result[keys[0]].value == "value"
    assert result[keys[1]].value == "value2"
    assert end == len(content)


# LLM-generated content at query #5
#--------------------------

```python
def test_TokenizingJSONObject_raises_error_when_nextchar_is_not_double_quote_after_comma_and_whitespace():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.positional_validation import JSONDecodeError
    import typesystem.tokenize.tokenize_json as module
    import json

    original_scanstring = module.scanstring
    original_WHITESPACE = module.WHITESPACE

    def mock_scanstring(s, end, strict):
        return "key", end + 5

    def mock_scan_once(s, end):
        token = ScalarToken("value", end, end + 4, "")
        return token, end + 5

    class MockMatch:
        def __init__(self, end_pos):
            self._end = end_pos

        def end(self):
            return self._end

    class MockWhitespace:
        def match(self, s, pos):
            return MockMatch(pos + 1)

    module.scanstring = mock_scanstring
    module.WHITESPACE = MockWhitespace()
    memo = {}
    content = '{"key": "value",  "key2": "value2"}'
    s = '{"key": "value",  "key2": "value2"}'
    end = 0
    try:
        result = _TokenizingJSONObject((s, end), True, mock_scan_once, memo, content)
        assert False, "Expected JSONDecodeError"
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)
    finally:
        module.scanstring = original_scanstring
        module.WHITESPACE = original_WHITESPACE


# LLM-generated content at query #6
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    content = '"hello"'
    context = type('Context', (), {'parse_string': lambda s, idx, strict: ('hello', len(s)), 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 'hello'
    assert token.string == '"hello"'
    assert end == len(content)

def test_make_scanner_returns_scalar_token_for_null():
    content = 'null'
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == len(content)

def test_make_scanner_returns_scalar_token_for_true():
    content = 'true'
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == len(content)

def test_make_scanner_returns_scalar_token_for_false():
    content = 'false'
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == len(content)

def test_make_scanner_returns_scalar_token_for_integer():
    content = '42'
    context = type('Context', (), {'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == len(content)

def test_make_scanner_returns_scalar_token_for_float():
    content = '3.14'
    context = type('Context', (), {'parse_float': float, 'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == len(content)

def test_make_scanner_returns_dict_token_for_object():
    content = '{"key": "value"}'
    parse_object = lambda string_idx, strict, scan_once, memo, content: ({ScalarToken('key', 1, 4, content): ScalarToken('value', 8, 14, content)}, len(content))
    context = type('Context', (), {'parse_object': parse_object, 'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {'key': 'value'}
    assert token.string == '{"key": "value"}'
    assert end == len(content)

def test_make_scanner_returns_list_token_for_array():
    content = '[1, 2, 3]'
    parse_array = lambda string_idx, scan_once: ([ScalarToken(1, 1, 1, content), ScalarToken(2, 4, 4, content), ScalarToken(3, 7, 7, content)], len(content))
    context = type('Context', (), {'parse_array': parse_array, 'memo': {}})()
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == '[1, 2, 3]'
    assert end == len(content)

def test_make_scanner_clears_memo_after_scan():
    content = '"test"'
    memo = {}
    context = type('Context', (), {'parse_string': lambda s, idx, strict: ('test', len(s)), 'strict': True, 'memo': memo})()
    scanner = _make_scanner(context, content)
    scanner(content, 0)
    assert memo == {}

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    content = 'invalid'
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #7
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 14, content)
    assert result == {key_token: value_token}
    assert end == 16

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, 6, 6, content), 7
        else:
            return ScalarToken(2, 14, 14, content), 15
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 14, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken("value", 11, 17, content)
    assert result == {key_token: value_token}
    assert end == 19

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 8, 14, content), 15
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
            return ScalarToken(1, 6, 6, content), 7
        else:
            return ScalarToken(2, 13, 13, content), 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 6, 12, content), 13
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


# LLM-generated content at query #8
#--------------------------

def test_make_scanner_with_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
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
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
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
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
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
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
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
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
    context = MockContext()
    content = "3.14"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert idx == 4

def test_make_scanner_with_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
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
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
    context = MockContext()
    content = "[]"
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert idx == 2

def test_make_scanner_with_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "hello", idx + 7
        def parse_array(self, string_idx, scan_once):
            return [], string_idx[1]
    context = MockContext()
    content = '"hello"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert idx == 7


# LLM-generated content at query #9
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
    content = '{"key": 123}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(123, idx, idx + 2, content)
        return token, idx + 3
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken(123, 7, 9, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 11

def test_TokenizingJSONObject_multiple_key_values():
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
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 15, 15, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == 17

def test_TokenizingJSONObject_with_whitespace():
    content = '{  "key"  :  42  }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(42, idx, idx + 1, content)
        return token, idx + 2
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 3, 6, content)
    value_token = ScalarToken(42, 12, 13, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == 18

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

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 6, content)
        return token, idx + 7
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 13, content)
    expected = {key_token: value_token}
    assert result == expected
    assert "key" in memo
    assert memo["key"] == "key"
    assert end == 15


# LLM-generated content at query #10
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0

def test_tokenize_json_null():
    token = tokenize_json("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 4

def test_tokenize_json_true():
    token = tokenize_json("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 4

def test_tokenize_json_false():
    token = tokenize_json("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 5

def test_tokenize_json_number_integer():
    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 2

def test_tokenize_json_number_float():
    token = tokenize_json("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 4

def test_tokenize_json_string():
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 7

def test_tokenize_json_empty_list():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 2

def test_tokenize_json_list_with_elements():
    token = tokenize_json("[1, true, null]")
    assert isinstance(token, ListToken)
    assert token.value == [1, True, None]
    assert token.string == "[1, true, null]"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 16

def test_tokenize_json_empty_dict():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 2

def test_tokenize_json_dict_with_elements():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 16

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"list": [1, 2], "nested": {"inner": true}}')
    assert isinstance(token, DictToken)
    assert token.value == {"list": [1, 2], "nested": {"inner": True}}
    assert token.string == '{"list": [1, 2], "nested": {"inner": true}}'
    assert token.start.line_no == 1
    assert token.start.column_no == 1

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'"bytes"')
    assert isinstance(token, ScalarToken)
    assert token.value == "bytes"
    assert token.string == '"bytes"'
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 1
    assert token.end.column_no == 7

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index >= 0

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start.line_no == 1
    assert token.start.column_no == 1
    assert token.end.line_no == 4
    assert token.end.column_no == 1

def test_tokenize_json_lookup_list():
    token = tokenize_json("[10, 20, 30]")
    child = token.lookup([1])
    assert isinstance(child, ScalarToken)
    assert child.value == 20
    assert child.string == "20"

def test_tokenize_json_lookup_dict():
    token = tokenize_json('{"a": 1, "b": 2}')
    child = token.lookup(["b"])
    assert isinstance(child, ScalarToken)
    assert child.value == 2
    assert child.string == "2"

def test_tokenize_json_lookup_key():
    token = tokenize_json('{"x": 100}')
    key_token = token.lookup_key(["x"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "x"
    assert key_token.string == '"x"'


# LLM-generated content at query #11
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    try:
        tokenize_json(b"")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    try:
        tokenize_json("   \n\t  ")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_scalar_null():
    token = tokenize_json("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_true():
    token = tokenize_json("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_false():
    token = tokenize_json("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_scalar_integer():
    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_scalar_float():
    token = tokenize_json("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_string():
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list_empty():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_list_with_elements():
    token = tokenize_json('[1, "two", false]')
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", False]
    assert token.string == '[1, "two", false]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_dict_empty():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_dict_with_items():
    token = tokenize_json('{"key": "value", "num": 42}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 42}
    assert token.string == '{"key": "value", "num": 42}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=28, char_index=27)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"list": [1, 2], "nested": {"inner": true}}')
    assert isinstance(token, DictToken)
    expected = {"list": [1, 2], "nested": {"inner": True}}
    assert token.value == expected
    assert token.string == '{"list": [1, 2], "nested": {"inner": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=48, char_index=47)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"test": "bytes"}')
    assert isinstance(token, DictToken)
    assert token.value == {"test": "bytes"}
    assert token.string == '{"test": "bytes"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_invalid_json_parse_error():
    try:
        tokenize_json('{"unclosed":')
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index == 12

def test_tokenize_json_invalid_json_unexpected_character():
    try:
        tokenize_json('{invalid}')
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index == 1

def test_tokenize_json_multiline_content():
    content = '{\n  "key": "value"\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == content
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=1, char_index=20)


# LLM-generated content at query #12
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
    token = tokenize_json('{"a": [1, {"b": true}]}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, {"b": True}]}
    assert token.string == '{"a": [1, {"b": true}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=24, char_index=23)

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
        assert e.position.char_index > 0

def test_tokenize_json_multiline():
    content = '{\n  "name": "John",\n  "age": 30\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.string == content
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=33)


# LLM-generated content at query #13
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
        parse_string = lambda self, string, idx, strict: ("", 0)
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
        parse_string = lambda self, string, idx, strict: ("", 0)
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
        parse_string = lambda self, string, idx, strict: ("", 0)
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
    token, idx = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert idx == 2

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
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
        def parse_array(self, string_idx, scan_once):
            return [ScalarToken(1, 1, 1, ""), ScalarToken(2, 3, 3, "")], 5
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
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
        def parse_object(self, string_idx, strict, scan_once, memo, content):
            key = ScalarToken("key", 1, 3, content)
            value = ScalarToken("value", 5, 9, content)
            return {key: value}, 11
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert idx == 11

def test_make_scanner_stop_iteration():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
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
        parse_float = float
        parse_int = int
        memo = {"key": "value"}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #14
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 14, content)
    assert result == {key_token: value_token}
    assert end == 16

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, 6, 6, content), 7
        else:
            return ScalarToken(2, 14, 14, content), 15
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 14, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken("value", 11, 17, content)
    assert result == {key_token: value_token}
    assert end == 19

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 8, 14, content), 15
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
            return ScalarToken(1, 6, 6, content), 7
        else:
            return ScalarToken(2, 14, 14, content), 15
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 7, 13, content), 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_scan_once_stop_iteration():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_32_evaluates_false():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json._make_scanner.__closure__[0].cell_contents
    content = "nullx"
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, content)
    result = scanner(content, 0)
    token = result[0]
    assert token.string == "null"
    assert token.value is None
    assert token.start.index == 0
    assert token.end.index == 3


# LLM-generated content at query #16
#--------------------------

def test_make_scanner_parse_object_not_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    import typesystem.tokenize.tokens
    context = typesystem.tokenize.tokenize_json.JSONTokenizer(strict=True)
    content = '{"key": "value"}'
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, typesystem.tokenize.tokens.DictToken)


# LLM-generated content at query #17
#--------------------------

```python
def test_make_scanner_parse_object_is_not_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json.Context()
    content = "{}"
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, content)
    assert scanner.__closure__[0].cell_contents != typesystem.tokenize.tokenize_json._TokenizingJSONObject


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    import typesystem.tokenize.tokenize_json as module
    original_scan_once = module.scan_once
    memo = {}
    content = '{"key": "value"}'
    s = content
    end = 0
    def mock_scan_once(s, idx):
        if s[idx] == '"':
            token, new_idx = module.scanstring(s, idx, True)
            return ScalarToken(token, idx, new_idx - 1, content), new_idx
        raise StopIteration(idx)
    module.scan_once = mock_scan_once
    result, new_end = _TokenizingJSONObject((s, end), True, mock_scan_once, memo, content)
    module.scan_once = original_scan_once
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"].value == "value"
    assert new_end == len(content)


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            token, new_idx = json.decoder.scanstring(s, idx+1, json.decoder.strict)
            return ScalarToken(token, idx, new_idx-1, content), new_idx
        raise StopIteration(idx)
    def whitespace_match(s, idx):
        class Match:
            def end(self):
                return idx
        return Match()
    result = _TokenizingJSONObject((content, 0), True, scan_once, memo, content, _w=whitespace_match, _ws=' \t\n\r')
    assert result[0] == {"key": ScalarToken("value", 8, 14, content)}
    assert result[1] == 16


# LLM-generated content at query #20
#--------------------------

def test_scalar_token_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_array = lambda self, *args: ([], 0)
        parse_string = lambda self, string, idx, strict: ("", idx)
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #21
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject(("{}", 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    expected = {}
    assert result == expected
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
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
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

def test_TokenizingJSONObject_key_not_string():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_nested_scan_once():
    content = '{"key": [1, 2]}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken([1, 2], idx, idx + 5, content), idx + 6
        return ScalarToken(None, idx, idx, content), idx + 1
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert call_count == 1
    assert end == len(content)

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value", "key": "value2"}'
    memo = {}
    def scan_once(s, idx):
        start = idx
        while s[idx] != '"':
            idx += 1
        idx += 1
        val = "value" if idx < 20 else "value2"
        return ScalarToken(val, start, idx - 1, content), idx
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 2
    keys = list(result.keys())
    assert keys[0].value == "key"
    assert keys[1].value == "key"
    assert keys[0] is keys[1]


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_whitespace_string():
    try:
        tokenize_json("   ")
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
    token = tokenize_json("[1, true, null]")
    assert isinstance(token, ListToken)
    assert token.value == [1, True, None]
    assert token.string == "[1, true, null]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

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
    assert token.value == {"list": [1, 2], "nested": {"inner": True}}
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=45, char_index=44)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'"bytes"')
    assert isinstance(token, ScalarToken)
    assert token.value == "bytes"
    assert token.string == '"bytes"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.char_index == 0

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=9)


# LLM-generated content at query #2
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    expected = {}
    assert result == expected
    assert new_end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 9, 15, content)
        return token, 16
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 9, 15, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 17

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    s = content
    end = 0
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, 6, 6, content)
            return token, 7
        else:
            token = ScalarToken(2, 14, 14, content)
            return token, 15
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 14, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert new_end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 13, 19, content)
        return token, 20
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 3, 6, content)
    value_token = ScalarToken("value", 13, 19, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 21

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 9, 15, content), 16
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    s = content
    end = 0
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            token = ScalarToken(1, 6, 6, content)
            return token, 7
        else:
            token = ScalarToken(2, 13, 13, content)
            return token, 14
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 7, 13, content), 14
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_stop_iteration_on_value():
    content = '{"key":'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #3
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(0)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
    assert end == 2

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 4, content)
        return token, idx + 5
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 3, content)
    value_token = ScalarToken("value", 8, 12, content)
    assert result == {key_token: value_token}
    assert end == 15

def test_TokenizingJSONObject_multiple_key_values():
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
    key_token_a = ScalarToken("a", 1, 1, content)
    key_token_b = ScalarToken("b", 8, 8, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 4, content)
        return token, idx + 5
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 4, content)
    value_token = ScalarToken("value", 11, 15, content)
    assert result == {key_token: value_token}
    assert end == 18

def test_TokenizingJSONObject_missing_colon_raises_error():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 5
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ':' delimiter"

def test_TokenizingJSONObject_missing_comma_raises_error():
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

def test_TokenizingJSONObject_missing_quote_on_key_raises_error():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 5
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_stop_iteration_raises_error():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


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

def test_make_scanner_dict():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_string = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '{"key": "value"}'
    def mock_parse_object(string_idx, strict, scan_once, memo, content):
        return {}, len(content)
    _TokenizingJSONObject = mock_parse_object
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.string == '{"key": "value"}'
    assert idx == len(content)

def test_make_scanner_list():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = lambda self, string_idx, scan_once: ([], len(content))
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '["item1", "item2"]'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.string == '["item1", "item2"]'
    assert idx == len(content)

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_float = float
        parse_int = int
        memo = {"key": "value"}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, idx = scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = None
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = 'invalid'
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration:
        pass


# LLM-generated content at query #5
#--------------------------

def test_scalar_token_null_equality_with_different_content():
    token1 = ScalarToken(None, 0, 3, "null")
    token2 = ScalarToken(None, 0, 3, "null")
    result = token1 == token2
    assert result == True
    token3 = ScalarToken(None, 0, 3, "null")
    token4 = ScalarToken(None, 0, 3, "NULL")
    result = token3 == token4
    assert result == False


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
        tokenize_json("   \n\t  ")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(line_no=1, column_no=1, char_index=0)

def test_tokenize_json_scalar_null():
    token = tokenize_json("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_true():
    token = tokenize_json("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_false():
    token = tokenize_json("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=5, char_index=4)

def test_tokenize_json_scalar_integer():
    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_scalar_float():
    token = tokenize_json("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == "3.14"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=4, char_index=3)

def test_tokenize_json_scalar_string():
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_list_empty():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_list_simple():
    token = tokenize_json("[1, 2, 3]")
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_dict_empty():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_dict_simple():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"a": [1, {"b": true}]}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, {"b": True}]}
    assert token.string == '{"a": [1, {"b": true}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=24, char_index=23)

def test_tokenize_json_bytes_input():
    token = tokenize_json(b'{"x": 5}')
    assert isinstance(token, DictToken)
    assert token.value == {"x": 5}
    assert token.string == '{"x": 5}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=8, char_index=7)

def test_tokenize_json_invalid_json():
    try:
        tokenize_json('{"unclosed":')
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting" in e.text

def test_tokenize_json_invalid_bytes():
    try:
        tokenize_json(b'{"invalid": \x80}')
    except ParseError as e:
        assert e.code == "parse_error"

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)


# LLM-generated content at query #7
#--------------------------

```python
def test_make_scanner_parse_object_is_not_TokenizingJSONObject():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import Token
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


# LLM-generated content at query #8
#--------------------------

def test_TokenizingJSONObject_raises_error_when_nextchar_not_double_quote_at_line_72():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.positional_validation import JSONDecodeError
    import typing
    content = '{"key": "value", "another": "value"}'
    s = '{"key": "value", "another": "value"}'
    end = len('{"key": "value", ')
    memo = {}
    def mock_scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 5
    def mock_whitespace_match(s, idx):
        class Match:
            def end(self):
                return idx
        return Match()
    WHITESPACE_STR = " \t\n\r"
    try:
        _TokenizingJSONObject((s, end), True, mock_scan_once, memo, content, _w=mock_whitespace_match, _ws=WHITESPACE_STR)
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #9
#--------------------------

def test_TokenizingJSONObject_empty_object():
    s = "{}"
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
    assert result == {}
    assert new_end == 2

def test_TokenizingJSONObject_simple_key_value():
    s = '{"key": 123}'
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx+2, content), idx+3
    result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
    key_token = list(result.keys())[0]
    value_token = result[key_token]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert value_token.value == 123
    assert new_end == len(s)

def test_TokenizingJSONObject_multiple_pairs():
    s = '{"a": 1, "b": 2}'
    end = 0
    strict = True
    memo = {}
    content = s
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, idx, idx, content), idx+1
        else:
            return ScalarToken(2, idx, idx, content), idx+1
    result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
    assert len(result) == 2
    assert result[ScalarToken("a", 1, 3, content)].value == 1
    assert result[ScalarToken("b", 9, 11, content)].value == 2
    assert new_end == len(s)

def test_TokenizingJSONObject_with_whitespace():
    s = '{ "key" : 123 }'
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx+2, content), idx+3
    result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
    key_token = list(result.keys())[0]
    assert key_token.string == '"key"'
    assert result[key_token].value == 123
    assert new_end == len(s)

def test_TokenizingJSONObject_missing_colon():
    s = '{"key" 123}'
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx+2, content), idx+3
    try:
        _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    s = '{"a": 1 "b": 2}'
    end = 0
    strict = True
    memo = {}
    content = s
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, idx, idx, content), idx+1
        else:
            return ScalarToken(2, idx, idx, content), idx+1
    try:
        _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_key_not_string():
    s = '{key: 123}'
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken(123, idx, idx+2, content), idx+3
    try:
        _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_stop_iteration_value():
    s = '{"key": }'
    end = 0
    strict = True
    memo = {}
    content = s
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), strict, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #10
#--------------------------

def test_nextchar_not_double_quote_raises_error():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.position import Position
    import json
    import typesystem.tokenize.tokenize_json as module
    memo = {}
    content = '{"key": "value", "another": 123}'
    def scan_once(s, idx):
        if s[idx] == '"':
            token, end = module.scanstring(s, idx, True)
            return ScalarToken(token, idx, end - 1, content), end
        elif s[idx].isdigit():
            start = idx
            while idx < len(s) and s[idx].isdigit():
                idx += 1
            value = int(s[start:idx])
            return ScalarToken(value, start, idx - 1, content), idx
        else:
            raise StopIteration(idx)
    s = '{"key": "value", "another": 123}'
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert len(result) == 2
    s = '{"key": "value", another: 123}'
    try:
        _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
        assert False
    except json.JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"
        assert e.pos == 16


# LLM-generated content at query #11
#--------------------------

```python
def test_scan_once_does_not_raise_stop_iteration():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json

    class MockScanner:
        def __init__(self, return_value):
            self.return_value = return_value
            self.called = False

        def __call__(self, s, end):
            self.called = True
            return self.return_value

    content = '{"key": "value"}'
    memo = {}
    scanner = MockScanner((ScalarToken("value", 7, 13, content), 14))
    result = _TokenizingJSONObject((content, 1), True, scanner, memo, content)
    assert scanner.called
    assert result[0] == {"key": "value"}
    assert result[1] == 14


# LLM-generated content at query #12
#--------------------------

def test_scalar_token_null_value():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import typesystem.tokenize.tokenize_json as module
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #13
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert result == {}
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
    assert result == {key_token: value_token}
    assert end == 17

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    def scan_once(s, idx):
        if idx == 7:
            token = ScalarToken(1, 7, 7, content)
            return token, 8
        else:
            token = ScalarToken(2, 14, 14, content)
            return token, 15
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 7, 7, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 14, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 13, 19, content)
        return token, 20
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken("value", 13, 19, content)
    assert result == {key_token: value_token}
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
    def scan_once(s, idx):
        if idx == 7:
            token = ScalarToken(1, 7, 7, content)
            return token, 8
        else:
            token = ScalarToken(2, 14, 14, content)
            return token, 15
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", 7, 13, content)
        return token, 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_stop_iteration_error():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #14
#--------------------------

```python
def test_null_token_creation():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    import json
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = json.decoder.scanstring
        parse_array = json.decoder.JSONArray
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_32_evaluates_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, "nullx")
    token, end = scanner("nullx", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4
    scanner2 = _make_scanner(context, "xnull")
    try:
        scanner2("xnull", 0)
        assert False
    except StopIteration:
        pass


# LLM-generated content at query #16
#--------------------------

def test_scalar_token_null_value():
    from typesystem.tokenize.tokens import ScalarToken
    token = ScalarToken(None, 0, 3, "null")
    result = token.value
    expected = None
    assert result == expected


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_53_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    s = content
    end = 0
    strict = True
    memo = {}
    def mock_scan_once(s, idx):
        return ScalarToken("value", 7, 13, content), 14
    result_dict, new_end = _TokenizingJSONObject((s, end), strict, mock_scan_once, memo, content)
    assert isinstance(result_dict, dict)
    assert "key" in result_dict
    assert result_dict["key"].value == "value"
    assert new_end == len(content)


# LLM-generated content at query #18
#--------------------------

def test_make_scanner_with_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_with_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = 'true'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_with_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = 'false'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_with_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = '42'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_with_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = '3.14'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_with_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}, 'parse_string': lambda s, idx, strict: ('hello', idx + 7)})()
    content = '"hello"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 'hello'
    assert token.string == '"hello"'
    assert end == 7

def test_make_scanner_with_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}, 'parse_object': lambda *args: ({}, 2)})()
    content = '{}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_with_empty_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}, 'parse_array': lambda *args: ([], 2)})()
    content = '[]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {'key': 'value'}})()
    content = 'null'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert context.memo == {}

def test_make_scanner_stop_iteration_on_invalid():
    from typesystem.tokenize.tokenize_json import _make_scanner
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    content = 'invalid'
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_61_evaluates_false():
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
    result, end_index = _TokenizingJSONObject((content, 0), True, scan_once, memo, content, _w=whitespace_match, _ws='')
    assert isinstance(result, dict)
    assert 'key' in result
    assert result['key'].value == 'value'


# LLM-generated content at query #20
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(1)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    expected = {}
    assert result == expected
    assert new_end == 1

def test_TokenizingJSONObject_single_key_value():
    content = '{"key": 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, 8, 10, content), 11
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken(123, 8, 10, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 11

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    s = content
    end = 0
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, 6, 7, content), 8
        else:
            return ScalarToken(2, 14, 15, content), 16
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 7, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 15, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert new_end == 16

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, 11, 13, content), 14
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken(123, 11, 13, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 14

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, 8, 10, content), 11
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ':' delimiter"

def test_TokenizingJSONObject_missing_comma():
    content = '{"a": 1 "b": 2}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(1, 6, 7, content), 8
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting ',' delimiter"

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(123, 6, 8, content), 9
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_stop_iteration_value():
    content = '{"key": }'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from unittest.mock import Mock, patch
    import json

    content = '{"key": "value"}'
    s = content
    end = 1
    memo = {}
    strict = True
    scan_once = Mock(return_value=(ScalarToken("value", 8, 14, content), 15))
    _w = Mock(side_effect=lambda s, idx: Mock(end=lambda: idx + 1) if s[idx] in " \t\n\r" else Mock(end=lambda: idx))
    _ws = " \t\n\r"

    with patch('typesystem.tokenize.tokenize_json.scanstring', return_value=("key", 6)):
        with patch('typesystem.tokenize.tokenize_json.WHITESPACE_STR', _ws):
            with patch('typesystem.tokenize.tokenize_json.WHITESPACE.match', _w):
                result, new_end = _TokenizingJSONObject((s, end), strict, scan_once, memo, content, _w, _ws)
    assert result == {"key": ScalarToken("value", 8, 14, content)}
    assert new_end == 16


# LLM-generated content at query #22
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
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value", start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    value_token = list(result.values())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_multiple_pairs():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        elif s[idx] == '2':
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 2
    assert result[ScalarToken("a", 1, 3, content)].value == 1
    assert result[ScalarToken("b", 9, 11, content)].value == 2
    assert call_count == 2
    assert end == len(content)

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value", start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    value_token = list(result.values())[0]
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert value_token.string == '"value"'
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx
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
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx + 1
        elif s[idx] == '2':
            token = ScalarToken(2, idx, idx, content)
            return token, idx + 1
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_key_not_string():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value", "key": "other"}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if s[idx] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            val = "value" if call_count == 1 else "other"
            token = ScalarToken(val, start, idx - 1, content)
            return token, idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.value == "key"
    assert "key" in memo
    assert end == len(content)


# LLM-generated content at query #23
#--------------------------

```python
def test_make_scanner_parse_object_not_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json.Context()
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, "")
    assert type(context.parse_object).__name__ != "_TokenizingJSONObject"


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_scalar_token_null_value():
    from typesystem.tokenize.tokens import ScalarToken
    token = ScalarToken(None, 0, 3, "null")
    result = token.value
    expected = None
    assert result == expected


# LLM-generated content at query #26
#--------------------------

```python
def test_make_scanner_parse_object_not_TokenizingJSONObject():
    from typesystem.tokenize.tokenize_json import _make_scanner
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


# LLM-generated content at query #27
#--------------------------

def test_null_token_creation():
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.tokenize_json import _make_scanner
    import typesystem.tokenize.tokenize_json as module
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    expected_token = ScalarToken(None, 0, 3, "null")
    assert token == expected_token
    assert end == 4


# LLM-generated content at query #28
#--------------------------

def test_make_scanner_scalar_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
        memo = {}
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
        memo = {}
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
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
    from typesystem.tokenize.tokens import DictToken, ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            if string[idx:idx+4] == '"key"':
                return "key", idx + 5
            return "value", idx + 7
        def parse_object(self, args, strict, scan_once, memo, content):
            return {ScalarToken("key", 1, 4, content): ScalarToken("value", 7, 12, content)}, 13
        parse_array = None
        parse_float = float
        parse_int = int
    context = MockContext()
    content = '{"key":"value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key":"value"}'
    assert end == 15

def test_make_scanner_list_token():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken, ScalarToken
    class MockContext:
        strict = True
        memo = {}
        def parse_string(self, string, idx, strict):
            return "item", idx + 6
        def parse_array(self, args, scan_once):
            string, idx = args
            token1, idx = scan_once(string, idx)
            token2, idx = scan_once(string, idx + 1)
            return [token1, token2], idx + 1
        parse_object = None
        parse_float = float
        parse_int = int
    context = MockContext()
    content = '["item1","item2"]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.string == '["item1","item2"]'
    assert end == 18

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        memo = {"key": "value"}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        memo = {}
        parse_string = None
        parse_array = None
        parse_object = None
        parse_float = float
        parse_int = int
    context = MockContext()
    content = 'invalid'
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #29
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = "{}"
    memo = {}
    result, end = _TokenizingJSONObject(("{}", 0), True, lambda s, idx: (ScalarToken(None, idx, idx, content), idx + 1), memo, content)
    expected = {}
    assert result == expected
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
    assert key_token.value == "key"
    assert key_token.string == '"key"'
    value_token = result[key_token]
    assert value_token.value == "value"
    assert value_token.string == '"value"'
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
    keys = list(result.keys())
    values = list(result.values())
    assert keys[0].value == "a"
    assert values[0].value == 1
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
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.value == "key"
    value_token = result[key_token]
    assert value_token.value == "value"
    assert end == len(content)

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 5
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

def test_TokenizingJSONObject_key_not_string():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 5
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
        if call_count[0] == 1:
            return ScalarToken("value", idx, idx + 5, content), idx + 6
        else:
            return ScalarToken("value2", idx, idx + 6, content), idx + 7
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert key_token.value == "key"
    value_token = result[key_token]
    assert value_token.value == "value2"
    assert "key" in memo


# LLM-generated content at query #30
#--------------------------

```python
def test_scalar_token_null_equality():
    token1 = ScalarToken(None, 0, 3, "null")
    token2 = ScalarToken(None, 0, 3, "null")
    result = token1 == token2
    assert result == True


