####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
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
        if s[idx:idx+1] == '"':
            start = idx
            idx += 1
            while s[idx] != '"':
                idx += 1
            idx += 1
            token = ScalarToken("value", start, idx-1, content)
            return token, idx
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 14, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 16

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
        elif call_count == 2:
            token = ScalarToken(2, 14, 14, content)
            return token, 15
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    key_token_b = ScalarToken("b", 9, 10, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 14, 14, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert new_end == 17

def test_TokenizingJSONObject_whitespace_around_colon():
    content = '{"key" : "value"}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        start = idx
        idx += 1
        while s[idx] != '"':
            idx += 1
        idx += 1
        token = ScalarToken("value", start, idx-1, content)
        return token, idx
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 10, 16, content)
    expected = {key_token: value_token}
    assert result == expected
    assert new_end == 18

def test_TokenizingJSONObject_missing_colon_raises_error():
    content = '{"key" "value"}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 8, 14, content), 15
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma_raises_error():
    content = '{"a": 1 "b": 2}'
    s = content
    end = 0
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScalarToken(1, 6, 6, content), 7
        elif call_count == 2:
            return ScalarToken(2, 14, 14, content), 15
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes_raises_error():
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

def test_TokenizingJSONObject_scan_once_stop_iteration_raises_error():
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
        assert "Expecting value" in str(e)


# LLM-generated content at query #2
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    context = type('Context', (), {'parse_string': lambda s, idx, strict: ('test', idx + 6), 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 'test'
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_returns_dict_token_for_object():
    parse_object = lambda string_idx, strict, scan, memo, content: ({}, string_idx[1] + 2)
    context = type('Context', (), {'parse_object': parse_object, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '{}')
    token, end = scanner('{}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_returns_list_token_for_array():
    parse_array = lambda string_idx, scan: ([], string_idx[1] + 2)
    context = type('Context', (), {'parse_array': parse_array, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '[]')
    token, end = scanner('[]', 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_returns_scalar_token_for_null():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_true():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_false():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_returns_scalar_token_for_integer():
    context = type('Context', (), {'parse_int': int, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '123')
    token, end = scanner('123', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == '123'
    assert end == 3

def test_make_scanner_returns_scalar_token_for_float():
    context = type('Context', (), {'parse_float': float, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '123.45')
    token, end = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == '123.45'
    assert end == 6

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'invalid')
    try:
        scanner('invalid', 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_clears_memo_after_scan():
    memo = {}
    parse_object = lambda string_idx, strict, scan, memo, content: ({}, string_idx[1] + 2)
    context = type('Context', (), {'parse_object': parse_object, 'strict': True, 'memo': memo})()
    scanner = _make_scanner(context, '{}')
    scanner('{}', 0)
    assert memo == {}


# LLM-generated content at query #3
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
        parse_array = lambda self, args, scan_once: ([], args[1] + 1)
        parse_string = lambda self, string, idx, strict: ("", idx)
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #4
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
    def scan_once(s, idx):
        if idx == 6:
            token = ScalarToken(1, 6, 6, content)
            return token, 7
        else:
            token = ScalarToken(2, 13, 13, content)
            return token, 14
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 9, 10, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == 16

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
    def scan_once(s, idx):
        if idx == 6:
            token = ScalarToken(1, 6, 6, content)
            return token, 7
        else:
            token = ScalarToken(2, 12, 12, content)
            return token, 13
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

def test_TokenizingJSONObject_stop_iteration_value():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #5
#--------------------------

def test_nextchar_not_comma_raises_error():
    import typesystem.tokenize.tokenize_json
    import json
    s = '{"key": "value" "another": "value2"}'
    content = s
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            token, end = json.decoder.scanstring(s, idx + 1, strict=True)
            return typesystem.tokenize.tokens.ScalarToken(token, idx, end - 1, content), end
        raise StopIteration(idx)
    try:
        typesystem.tokenize.tokenize_json._TokenizingJSONObject((s, 1), True, scan_once, memo, content)
    except json.JSONDecodeError as e:
        pass
    else:
        assert False, "Expected JSONDecodeError"


# LLM-generated content at query #6
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    context = type('Context', (), {'parse_string': lambda s, i, strict: ('test', i + 4), 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 'test'
    assert token.string == '"test"'
    assert end == 5

def test_make_scanner_returns_dict_token_for_object():
    def parse_object(args, strict, scan_once, memo, content):
        return {ScalarToken('key', 1, 3, content): ScalarToken('value', 6, 10, content)}, 12
    context = type('Context', (), {'parse_object': parse_object, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '{"key":"value"}')
    token, end = scanner('{"key":"value"}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {'key': 'value'}
    assert token.string == '{"key":"value"}'
    assert end == 13

def test_make_scanner_returns_list_token_for_array():
    def parse_array(args, scan_once):
        return [ScalarToken(1, 1, 1, '[1]'), ScalarToken(2, 3, 3, '[1,2]')], 5
    context = type('Context', (), {'parse_array': parse_array, 'memo': {}})()
    scanner = _make_scanner(context, '[1,2]')
    token, end = scanner('[1,2]', 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[1,2]'
    assert end == 5

def test_make_scanner_returns_scalar_token_for_null():
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_true():
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert end == 4

def test_make_scanner_returns_scalar_token_for_false():
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

def test_make_scanner_returns_scalar_token_for_integer():
    context = type('Context', (), {'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, '42')
    token, end = scanner('42', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'
    assert end == 2

def test_make_scanner_returns_scalar_token_for_float():
    context = type('Context', (), {'parse_float': float, 'memo': {}})()
    scanner = _make_scanner(context, '3.14')
    token, end = scanner('3.14', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_clears_memo_after_scan():
    context = type('Context', (), {'parse_string': lambda s, i, strict: ('test', i + 4), 'strict': True, 'memo': {'key': 'value'}})()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert context.memo == {}

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    context = type('Context', (), {'memo': {}})()
    scanner = _make_scanner(context, 'invalid')
    try:
        scanner('invalid', 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_null_token_creation():
    content = "null"
    token, end = ScalarToken(None, 0, 3, content), 4
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert token.start.line == 1
    assert token.start.column == 1
    assert token.end.line == 1
    assert token.end.column == 4
    assert end == 4


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_24_evaluates_to_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    import json
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_array = json.JSONDecoder().scan_once
        parse_string = json.decoder.scanstring
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start.line == 1
    assert token.start.column == 1
    assert token.end.line == 1
    assert token.end.column == 16


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_53_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    s = '{"key": "value"}'
    content = s
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
    result, end = _TokenizingJSONObject((s, 0), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"].value == "value"
    assert end == len(s)


# LLM-generated content at query #10
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

def test_tokenize_json_bytes():
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'

def test_tokenize_json_simple_object():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'

def test_tokenize_json_simple_array():
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == '[1, 2, 3]'

def test_tokenize_json_scalar_true():
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'

def test_tokenize_json_scalar_false():
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'

def test_tokenize_json_scalar_null():
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'

def test_tokenize_json_scalar_number():
    token = tokenize_json('42')
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == '42'

def test_tokenize_json_scalar_float():
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'

def test_tokenize_json_nested_object():
    token = tokenize_json('{"a": {"b": 1}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": 1}}
    assert token.string == '{"a": {"b": 1}}'

def test_tokenize_json_nested_array():
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    assert token.string == '[[1, 2], [3, 4]]'

def test_tokenize_json_parse_error():
    try:
        tokenize_json('{"key": "value"')
    except ParseError as exc:
        assert exc.code == "parse_error"

def test_tokenize_json_token_start_end():
    token = tokenize_json('{"key": "value"}')
    assert token.start.char_index == 0
    assert token.end.char_index == len('{"key": "value"}') - 1

def test_tokenize_json_token_lookup():
    token = tokenize_json('{"key": "value"}')
    child = token.lookup(["key"])
    assert isinstance(child, ScalarToken)
    assert child.value == "value"

def test_tokenize_json_token_lookup_key():
    token = tokenize_json('{"key": "value"}')
    key_token = token.lookup_key(["key"])
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"


# LLM-generated content at query #11
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
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

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
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

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
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

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
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_dict():
    from typesystem.tokenize.tokenize_json import _make_scanner, _TokenizingJSONObject
    from typesystem.tokenize.tokens import DictToken, ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            if string[idx:].startswith('"key"'):
                return "key", idx + 5
            return "value", idx + 7
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert end == 16

def test_make_scanner_list():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken, ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "item", idx + 6
        def parse_array(self, args, scan_once):
            string, idx = args
            items = []
            while string[idx] != ']':
                token, idx = scan_once(string, idx)
                items.append(token)
                if string[idx] == ',':
                    idx += 1
            return items, idx + 1
    context = MockContext()
    content = '["item"]'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == ["item"]
    assert token.string == '["item"]'
    assert end == 8

def test_make_scanner_stop_iteration():
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

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {'key': 'value'}
        def parse_string(self, string, idx, strict):
            return "test", idx + 6
        def parse_array(self, args, scan_once):
            pass
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}


# LLM-generated content at query #12
#--------------------------

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


# LLM-generated content at query #13
#--------------------------

```python
def test_null_token_creation():
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.tokenize_json import _make_scanner
    import json

    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_array = lambda self, args, scan_once: ([], args[1] + 1)
        parse_string = lambda self, string, idx, strict: ("", idx)

    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    from unittest.mock import Mock, patch
    import json

    content = '{"key": "value"}'
    memo = {}
    scan_once = Mock(side_effect=lambda s, idx: (ScalarToken("value", 9, 15, content), 16))
    s = content
    end = 0
    result = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    assert result[0] == {"key": "value"}
    assert result[1] == 17


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken, DictToken, ListToken
    import json
    class MockContext:
        parse_array = json.JSONDecoder().scan_once
        parse_string = json.decoder.scanstring
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    content = '{"key": "value"}'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token._value is not None
    assert token._value != _TokenizingJSONObject


# LLM-generated content at query #16
#--------------------------

def test_tokenize_json_empty_string():
    content = ""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_whitespace_string():
    content = "   "
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

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

def test_tokenize_json_integer():
    content = "42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
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

def test_tokenize_json_empty_list():
    content = "[]"
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_list_with_elements():
    content = '[1, "two", false]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", False]
    assert token.string == '[1, "two", false]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_empty_dict():
    content = "{}"
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_dict_with_elements():
    content = '{"key": "value", "num": 42}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 42}
    assert token.string == '{"key": "value", "num": 42}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=28, char_index=27)

def test_tokenize_json_nested_structure():
    content = '{"list": [1, 2, 3], "nested": {"inner": true}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    expected = {"list": [1, 2, 3], "nested": {"inner": True}}
    assert token.value == expected
    assert token.string == '{"list": [1, 2, 3], "nested": {"inner": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=48, char_index=47)

def test_tokenize_json_bytes_input():
    content = b'"hello"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.string == '"hello"'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)

def test_tokenize_json_invalid_json():
    content = '{"key": "value"'
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index > 0

def test_tokenize_json_multiline():
    content = '{\n  "key": "value"\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{\n  "key": "value"\n}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=1, char_index=20)


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_32_false():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json._make_scanner.__closure__[0].cell_contents
    content = ""
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, content)
    string = "x"
    idx = 0
    try:
        scanner(string, idx)
    except StopIteration:
        pass
    else:
        assert False


# LLM-generated content at query #18
#--------------------------

def test_nextchar_not_double_quote_raises_error():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value", "another": 123}'
    memo = {}
    def mock_scan_once(s, idx):
        if s[idx] == '"':
            end = s.find('"', idx + 1)
            token = ScalarToken(s[idx + 1:end], idx, end, content)
            return token, end + 1
        elif s[idx].isdigit():
            end = idx + 1
            while end < len(s) and s[end].isdigit():
                end += 1
            token = ScalarToken(int(s[idx:end]), idx, end - 1, content)
            return token, end
        else:
            raise StopIteration(idx)
    s = content
    end = 0
    s_and_end = (s, end)
    strict = True
    result, new_end = _TokenizingJSONObject(s_and_end, strict, mock_scan_once, memo, content)
    assert result == {"key": "value", "another": 123}
    assert new_end == len(content)
    invalid_content = '{"key": "value", invalid: 123}'
    s = invalid_content
    end = 0
    s_and_end = (s, end)
    try:
        _TokenizingJSONObject(s_and_end, strict, mock_scan_once, memo, invalid_content)
    except json.JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"
        assert e.pos == 16
    else:
        assert False, "Expected JSONDecodeError"


# LLM-generated content at query #19
#--------------------------

def test_tokenize_json_empty_string():
    content = ""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    content = b""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    content = "   \n\t  "
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

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

def test_tokenize_json_integer():
    content = "42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
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

def test_tokenize_json_empty_object():
    content = "{}"
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_object():
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_empty_array():
    content = "[]"
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_array():
    content = '[1, "two", false]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", False]
    assert token.string == '[1, "two", false]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

def test_tokenize_json_nested_structure():
    content = '{"a": [{"b": 2}]}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"a": [{"b": 2}]}
    assert token.string == '{"a": [{"b": 2}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=18, char_index=17)

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
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index > 0

def test_tokenize_json_invalid_bytes():
    content = b'{"invalid": \x80}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert "invalid" in token.value

def test_tokenize_json_multiline():
    content = '{\n  "key": "value"\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{\n  "key": "value"\n}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=1, char_index=20)


# LLM-generated content at query #20
#--------------------------

def test_make_scanner_returns_scalar_token_for_string():
    context = type('Context', (), {'parse_string': lambda s, i, strict: ('test', i + 6), 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '"test"')
    token, idx = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 'test'
    assert token.string == '"test"'
    assert idx == 6

def test_make_scanner_returns_dict_token_for_object():
    def parse_object(args, strict, scan_once, memo, content):
        return {ScalarToken('key', 1, 4, content): ScalarToken('value', 7, 12, content)}, 13
    context = type('Context', (), {'parse_object': parse_object, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '{"key":"value"}')
    token, idx = scanner('{"key":"value"}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {'key': 'value'}
    assert token.string == '{"key":"value"}'
    assert idx == 13

def test_make_scanner_returns_list_token_for_array():
    def parse_array(args, scan_once):
        return [ScalarToken(1, 1, 2, '[1]'), ScalarToken(2, 4, 5, '[1,2]')], 6
    context = type('Context', (), {'parse_array': parse_array, 'memo': {}})()
    scanner = _make_scanner(context, '[1,2]')
    token, idx = scanner('[1,2]', 0)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[1,2]'
    assert idx == 6

def test_make_scanner_returns_scalar_token_for_null():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'null')
    token, idx = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert idx == 4

def test_make_scanner_returns_scalar_token_for_true():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'true')
    token, idx = scanner('true', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == 'true'
    assert idx == 4

def test_make_scanner_returns_scalar_token_for_false():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'false')
    token, idx = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert idx == 5

def test_make_scanner_returns_scalar_token_for_integer():
    context = type('Context', (), {'parse_int': int, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '123')
    token, idx = scanner('123', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == '123'
    assert idx == 3

def test_make_scanner_returns_scalar_token_for_float():
    context = type('Context', (), {'parse_float': float, 'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, '123.45')
    token, idx = scanner('123.45', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.string == '123.45'
    assert idx == 6

def test_make_scanner_raises_stop_iteration_on_invalid_input():
    context = type('Context', (), {'strict': True, 'memo': {}})()
    scanner = _make_scanner(context, 'invalid')
    try:
        scanner('invalid', 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0

def test_make_scanner_clears_memo_after_scan():
    memo = {}
    context = type('Context', (), {'parse_string': lambda s, i, strict: ('test', i + 6), 'strict': True, 'memo': memo})()
    scanner = _make_scanner(context, '"test"')
    token, idx = scanner('"test"', 0)
    assert memo == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_parse_object_is_not_TokenizingJSONObject():
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
    parse_object = scanner.__closure__[0].cell_contents
    assert parse_object is not _TokenizingJSONObject


# LLM-generated content at query #22
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
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "test"
    assert token.string == '"test"'
    assert end == 6

def test_make_scanner_scalar_null():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    scanner = _make_scanner(context, 'null')
    token, end = scanner('null', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == 'null'
    assert end == 4

def test_make_scanner_scalar_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    scanner = _make_scanner(context, 'true')
    token, end = scanner('true', 0)
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
    scanner = _make_scanner(context, 'false')
    token, end = scanner('false', 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == 'false'
    assert end == 5

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
    scanner = _make_scanner(context, '42')
    token, end = scanner('42', 0)
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
    scanner = _make_scanner(context, '3.14')
    token, end = scanner('3.14', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.string == '3.14'
    assert end == 4

def test_make_scanner_dict():
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
    scanner = _make_scanner(context, '{}')
    token, end = scanner('{}', 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == '{}'
    assert end == 2

def test_make_scanner_list():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = None
        parse_array = lambda self, string_idx, scan_once: ([], 2)
        memo = {}
    context = MockContext()
    scanner = _make_scanner(context, '[]')
    token, end = scanner('[]', 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == '[]'
    assert end == 2

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_array = None
        memo = {"key": "value"}
    context = MockContext()
    scanner = _make_scanner(context, '"test"')
    token, end = scanner('"test"', 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = None
        parse_array = None
        memo = {}
    context = MockContext()
    scanner = _make_scanner(context, '')
    try:
        scanner('', 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #23
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
        if call_count == 0:
            token = ScalarToken(1, 6, 7, content)
            call_count += 1
            return token, 8
        else:
            token = ScalarToken(2, 14, 15, content)
            return token, 16
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 2, content)
    value_token_1 = ScalarToken(1, 6, 7, content)
    key_token_b = ScalarToken("b", 10, 11, content)
    value_token_2 = ScalarToken(2, 14, 15, content)
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
        if call_count == 0:
            token = ScalarToken(1, 6, 7, content)
            call_count += 1
            return token, 8
        else:
            token = ScalarToken(2, 14, 15, content)
            return token, 16
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

def test_TokenizingJSONObject_stop_iteration_value():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #24
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

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(123, idx, idx + 2, content)
        return token, idx + 3
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken(123, 9, 11, content)
    assert result == {key_token: value_token}
    assert end == 14

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
    assert result == {key_token: value_token}
    assert "key" in memo
    assert memo["key"] == "key"
    assert end == 15


# LLM-generated content at query #25
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    import typesystem.tokenize.tokenize_json as module
    original_scan_once = module.scan_once
    memo = {}
    content = '{"key": "value"}'
    def mock_scan_once(s, idx):
        if s[idx] == '"':
            end = idx + 6
            token = ScalarToken("value", idx, end - 1, content)
            return token, end
        raise StopIteration(idx)
    module.scan_once = mock_scan_once
    s = '{"key": "value"}'
    end = 0
    result, new_end = _TokenizingJSONObject((s, end), True, mock_scan_once, memo, content)
    module.scan_once = original_scan_once
    assert isinstance(result, dict)
    assert len(result) == 1
    key_token = list(result.keys())[0]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    value_token = result[key_token]
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    assert new_end == len(s)


# LLM-generated content at query #26
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    assert result == {}
    assert new_end == 2

def test_TokenizingJSONObject_simple_key_value():
    content = '{"key": 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken(123, 8, 10, content)
    assert result == {key_token: value_token}
    assert new_end == 12

def test_TokenizingJSONObject_multiple_pairs():
    content = '{"a": 1, "b": 2}'
    s = content
    end = 0
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        call_count += 1
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx+1
        elif s[idx] == '2':
            token = ScalarToken(2, idx, idx, content)
            return token, idx+1
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_a = ScalarToken("a", 1, 2, content)
    val_a = ScalarToken(1, 6, 6, content)
    key_b = ScalarToken("b", 9, 10, content)
    val_b = ScalarToken(2, 14, 14, content)
    assert result == {key_a: val_a, key_b: val_b}
    assert new_end == 16
    assert call_count == 2

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '1':
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken(123, 10, 12, content)
    assert result == {key_token: value_token}
    assert new_end == 15

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
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
        if s[idx] == '1':
            token = ScalarToken(1, idx, idx, content)
            return token, idx+1
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_key_quotes():
    content = '{key: 123}'
    s = content
    end = 0
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

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
        assert "Expecting value" in str(e)


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

def test_tokenize_json_empty_array():
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_array_with_elements():
    token = tokenize_json("[1, true, null]")
    assert isinstance(token, ListToken)
    assert token.value == [1, True, None]
    assert token.string == "[1, true, null]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_empty_object():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_object_with_key_value():
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_nested_structure():
    token = tokenize_json('{"a": [1, 2], "b": {"c": true}}')
    assert isinstance(token, DictToken)
    assert token.value == {"a": [1, 2], "b": {"c": True}}
    assert token.string == '{"a": [1, 2], "b": {"c": true}}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=33, char_index=32)

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
        assert isinstance(e.position, Position)

def test_tokenize_json_multiline():
    token = tokenize_json('[\n  1,\n  2\n]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2]
    assert token.string == '[\n  1,\n  2\n]'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=4, column_no=1, char_index=10)


# LLM-generated content at query #2
#--------------------------

def test_TokenizingJSONObject_empty_object():
    content = '{}'
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
    key_token = ScalarToken("key", 1, 3, content)
    value_token = ScalarToken("value", 8, 14, content)
    assert result == {key_token: value_token}
    assert end == 16

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return ScalarToken(1, 6, 6, content), 7
        else:
            call_count += 1
            return ScalarToken(2, 13, 13, content), 14
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 1, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    key_token_b = ScalarToken("b", 10, 10, content)
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
    key_token = ScalarToken("key", 2, 4, content)
    value_token = ScalarToken("value", 11, 17, content)
    assert result == {key_token: value_token}
    assert end == 19

def test_TokenizingJSONObject_missing_colon():
    content = '{"key" "value"}'
    memo = {}
    def scan_once(s, idx):
        return None, idx
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
        if call_count == 0:
            call_count += 1
            return ScalarToken(1, 6, 6, content), 7
        else:
            call_count += 1
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
        return None, idx
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


# LLM-generated content at query #3
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
        assert "Expecting ':' delimiter" in str(e)

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
            token = ScalarToken(2, 15, 15, content)
            return token, 16
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_key_not_string():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken("value", 7, 13, content), 14
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_stop_iteration():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #4
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
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
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
    value_token_1 = ScalarToken(1, 7, 7, content)
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


# LLM-generated content at query #5
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
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4

def test_make_scanner_with_true():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.string == "true"
    assert end == 4

def test_make_scanner_with_false():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.string == "false"
    assert end == 5

def test_make_scanner_with_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "123"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.string == "123"
    assert end == 3

def test_make_scanner_with_float():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "12.34"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34
    assert token.string == "12.34"
    assert end == 5

def test_make_scanner_with_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("parsed", idx + 8)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = '"string"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "parsed"
    assert token.string == '"string"'
    assert end == 9

def test_make_scanner_with_empty_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    def mock_parse_object(args, strict, scan, memo, content):
        return {}, args[1]
    MockContext.parse_object = staticmethod(mock_parse_object)
    context = MockContext()
    content = "{}"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert end == 2

def test_make_scanner_with_empty_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "[]"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert end == 2

def test_make_scanner_memo_cleared():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {"key": "value"}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration_on_invalid():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        parse_string = lambda self, s, idx, strict: ("", idx)
        parse_array = lambda self, args, scan: ([], args[1])
    context = MockContext()
    content = "invalid"
    scanner = _make_scanner(context, content)
    try:
        scanner(content, 0)
        assert False
    except StopIteration as e:
        assert e.args[0] == 0


# LLM-generated content at query #6
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
    value_token_1 = ScalarToken(1, 5, 5, content)
    key_token_b = ScalarToken("b", 9, 10, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result == {key_token_a: value_token_1, key_token_b: value_token_2}
    assert end == 15

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken(123, idx, idx + 2, content)
        return token, idx + 3
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 5, content)
    value_token = ScalarToken(123, 9, 11, content)
    assert result == {key_token: value_token}
    assert end == 14

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

def test_TokenizingJSONObject_key_not_string():
    content = '{123: "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 6, content)
        return token, idx + 7
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting property name enclosed in double quotes"

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 6, content)
        return token, idx + 7
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 4, content)
    value_token = ScalarToken("value", 8, 13, content)
    assert result == {key_token: value_token}
    assert "key" in memo
    assert memo["key"] == "key"

def test_TokenizingJSONObject_stop_iteration():
    content = '{"key":'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert e.msg == "Expecting value"


# LLM-generated content at query #7
#--------------------------

```python
def test_scalar_token_null_value():
    from typesystem.tokenize.tokens import ScalarToken
    token = ScalarToken(None, 0, 3, "null")
    result = token.value
    assert result is None


# LLM-generated content at query #8
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
        if s[idx:].startswith('123'):
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 1, 4, content)
    value_token = ScalarToken(123, 8, 10, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == len(content)

def test_TokenizingJSONObject_multiple_key_values():
    content = '{"a": 1, "b": 2}'
    memo = {}
    call_count = 0
    def scan_once(s, idx):
        nonlocal call_count
        if s[idx:].startswith('1'):
            token = ScalarToken(1, idx, idx, content)
            call_count += 1
            return token, idx+1
        if s[idx:].startswith('2'):
            token = ScalarToken(2, idx, idx, content)
            call_count += 1
            return token, idx+1
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token_a = ScalarToken('a', 1, 2, content)
    key_token_b = ScalarToken('b', 8, 9, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    expected = {key_token_a: value_token_1, key_token_b: value_token_2}
    assert result == expected
    assert end == len(content)
    assert call_count == 2

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : 123 }'
    memo = {}
    def scan_once(s, idx):
        if s[idx:].startswith('123'):
            token = ScalarToken(123, idx, idx+2, content)
            return token, idx+3
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 2, 5, content)
    value_token = ScalarToken(123, 10, 12, content)
    expected = {key_token: value_token}
    assert result == expected
    assert end == len(content)

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
        if s[idx:].startswith('1'):
            token = ScalarToken(1, idx, idx, content)
            call_count += 1
            return token, idx+1
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_key_not_string():
    content = '{123: "value"}'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_memoization():
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx:].startswith('"value"'):
            token = ScalarToken('value', idx, idx+6, content)
            return token, idx+7
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    key_token = ScalarToken('key', 1, 4, content)
    value_token = ScalarToken('value', 8, 13, content)
    expected = {key_token: value_token}
    assert result == expected
    assert 'key' in memo
    assert memo['key'] == 'key'
    assert end == len(content)


# LLM-generated content at query #9
#--------------------------

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
    expected_token = ScalarToken(None, 0, 3, content)
    assert token == expected_token
    assert end == 4


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_61_evaluates_to_true():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json
    content = '{"key": "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx] == '"':
            token, new_idx = json.decoder.scanstring(s, idx + 1, json.decoder.strict)
            return ScalarToken(token, idx, new_idx - 1, content), new_idx
        raise StopIteration(idx)
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"].value == "value"
    assert end == len(content)


# LLM-generated content at query #11
#--------------------------

```python
def test_whitespace_after_colon_handles_multiple_whitespace_characters():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json

    content = '{"key":  "value"}'
    memo = {}
    result, end_index = _TokenizingJSONObject((content, 1), True, lambda s, idx: (ScalarToken("value", 10, 14, content), 15), memo, content)
    key_token = result["key"]
    assert isinstance(key_token, ScalarToken)
    assert key_token.string == '"key"'
    assert key_token.value == "key"
    assert key_token.start.line == 1
    assert key_token.start.column == 2
    assert key_token.end.line == 1
    assert key_token.end.column == 5


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

def test_make_scanner_parse_object_is_not_TokenizingJSONObject():
    import typesystem.tokenize.tokenize_json
    context = typesystem.tokenize.tokenize_json.Context()
    content = ""
    scanner = typesystem.tokenize.tokenize_json._make_scanner(context, content)
    parse_object = typesystem.tokenize.tokenize_json._TokenizingJSONObject
    assert parse_object is not typesystem.tokenize.tokenize_json._TokenizingJSONObject


# LLM-generated content at query #14
#--------------------------

def test_TokenizingJSONObject_empty_object():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    s = "{}"
    end = 0
    memo = {}
    content = s
    def scan_once(s, idx):
        raise StopIteration(idx)
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    assert result == {}
    assert new_end == 2

def test_TokenizingJSONObject_single_key_value():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    s = '{"key": "value"}'
    end = 0
    memo = {}
    content = s
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 4, content)
        return token, idx + 6
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 3, content)
    value_token = ScalarToken("value", 8, 12, content)
    assert list(result.keys())[0] == key_token
    assert result[key_token] == value_token
    assert new_end == 15

def test_TokenizingJSONObject_multiple_key_values():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    s = '{"a": 1, "b": 2}'
    end = 0
    memo = {}
    content = s
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
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token_a = ScalarToken("a", 1, 1, content)
    key_token_b = ScalarToken("b", 8, 8, content)
    value_token_1 = ScalarToken(1, 6, 6, content)
    value_token_2 = ScalarToken(2, 13, 13, content)
    assert result[key_token_a] == value_token_1
    assert result[key_token_b] == value_token_2
    assert new_end == 16

def test_TokenizingJSONObject_with_whitespace():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    s = '{ "key" : "value" }'
    end = 0
    memo = {}
    content = s
    def scan_once(s, idx):
        token = ScalarToken("value", idx, idx + 4, content)
        return token, idx + 6
    result, new_end = _TokenizingJSONObject((s, end), True, scan_once, memo, content)
    key_token = ScalarToken("key", 2, 4, content)
    value_token = ScalarToken("value", 10, 14, content)
    assert list(result.keys())[0] == key_token
    assert result[key_token] == value_token
    assert new_end == 18

def test_TokenizingJSONObject_missing_colon():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokenize_json import JSONDecodeError
    s = '{"key" "value"}'
    end = 0
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 6
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ':' delimiter" in str(e)

def test_TokenizingJSONObject_missing_comma():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokenize_json import JSONDecodeError
    s = '{"a": 1 "b": 2}'
    end = 0
    memo = {}
    content = s
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
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_quote():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokenize_json import JSONDecodeError
    s = '{key: "value"}'
    end = 0
    memo = {}
    content = s
    def scan_once(s, idx):
        return ScalarToken("value", idx, idx + 4, content), idx + 6
    try:
        _TokenizingJSONObject((s, end), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)


# LLM-generated content at query #15
#--------------------------

def test_null_token_creation():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    context = type('Context', (), {'strict': True, 'parse_float': float, 'parse_int': int, 'memo': {}})()
    scanner = _make_scanner(context, "null")
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_scalar_token_null_equality():
    token1 = ScalarToken(None, 0, 3, "null")
    token2 = ScalarToken(None, 0, 3, "null")
    result = token1 == token2
    assert result


# LLM-generated content at query #18
#--------------------------

def test_make_scanner_with_empty_string():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = ""
    scanner = _make_scanner(context, content)
    token, end = scanner("", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = "null"
    scanner = _make_scanner(context, content)
    token, end = scanner("null", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert end == 4

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = "true"
    scanner = _make_scanner(context, content)
    token, end = scanner("true", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert end == 4

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = "false"
    scanner = _make_scanner(context, content)
    token, end = scanner("false", 0)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert end == 5

def test_make_scanner_with_number():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
    class MockContext:
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
        def parse_string(self, string, idx, strict):
            return "", idx
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = "42"
    scanner = _make_scanner(context, content)
    token, end = scanner("42", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert end == 2

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = "3.14"
    scanner = _make_scanner(context, content)
    token, end = scanner("3.14", 0)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert end == 4

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
    context = MockContext()
    content = '"hello"'
    scanner = _make_scanner(context, content)
    token, end = scanner('"hello"', 0)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert end == 7

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1]
        def parse_object(self, idx_strict_scan_once_memo_content):
            return {}, idx_strict_scan_once_memo_content[0][1] + 1
    context = MockContext()
    content = "{}"
    scanner = _make_scanner(context, content)
    token, end = scanner("{}", 0)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert end == 2

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
        def parse_array(self, idx_scan_once):
            return [], idx_scan_once[1] + 1
    context = MockContext()
    content = "[]"
    scanner = _make_scanner(context, content)
    token, end = scanner("[]", 0)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert end == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_61_evaluates_to_false():
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    import json
    import typesystem.tokenize.tokenize_json as module

    class MockScanOnce:
        def __init__(self, return_value):
            self.return_value = return_value
            self.calls = []

        def __call__(self, s, end):
            self.calls.append((s, end))
            return self.return_value

    content = '{"key": "value"}'
    memo = {}
    scan_once = MockScanOnce((ScalarToken("value", 9, 15, content), 16))
    result = _TokenizingJSONObject((content, 1), True, scan_once, memo, content)
    assert result[0] == {"key": "value"}
    assert result[1] == 17


# LLM-generated content at query #20
#--------------------------

```python
def test_whitespace_after_colon_skips_multiple_whitespace():
    from typesystem.tokenize.tokenize_json import _TokenizingJSONObject
    from typesystem.tokenize.tokens import ScalarToken
    import json

    content = '{"key":  "value"}'
    memo = {}
    def scan_once(s, idx):
        if s[idx:].startswith('"value"'):
            token = ScalarToken("value", idx, idx + 6, content)
            return token, idx + 7
        raise StopIteration(idx)

    result, end = _TokenizingJSONObject((content, 1), True, scan_once, memo, content)
    key_token = ScalarToken("key", 1, 5, content)
    assert result == {"key": ScalarToken("value", 10, 16, content)}
    assert end == 18


# LLM-generated content at query #21
#--------------------------

def test_scalar_token_null_value():
    token = ScalarToken(None, 0, 3, "null")
    result = token._get_value()
    assert result is None


# LLM-generated content at query #22
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

def test_make_scanner_scalar_integer():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ScalarToken
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

def test_make_scanner_object():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import DictToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
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

def test_make_scanner_array():
    from typesystem.tokenize.tokenize_json import _make_scanner
    from typesystem.tokenize.tokens import ListToken
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
        parse_float = float
        parse_int = int
        memo = {}
    def mock_parse_array(string_idx, scan_once):
        return [], 2
    MockContext.parse_array = staticmethod(mock_parse_array)
    context = MockContext()
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
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("test", idx + 6)
        parse_float = float
        parse_int = int
        memo = {'key': 'value'}
    context = MockContext()
    content = '"test"'
    scanner = _make_scanner(context, content)
    token, end = scanner(content, 0)
    assert context.memo == {}

def test_make_scanner_stop_iteration_on_invalid():
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        strict = True
        parse_string = lambda self, string, idx, strict: ("", 0)
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


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_48_evaluates_to_false():
    s = '{"key": "value"}'
    end = 7
    _ws = " \t\n\r"
    _w = lambda s, idx: type('Match', (), {'end': lambda: idx})()
    try:
        if s[end] in _ws:
            end += 1
            if s[end] in _ws:
                end = _w(s, end + 1).end()
    except IndexError:
        pass
    assert s[end] not in _ws


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_32_evaluates_false():
    from typesystem.tokenize.tokens import ScalarToken
    from typesystem.tokenize.tokenize_json import _make_scanner
    class MockContext:
        parse_array = lambda self, *args: ([], 0)
        parse_string = lambda self, *args: ("", 0)
        strict = True
        parse_float = float
        parse_int = int
        memo = {}
    context = MockContext()
    scanner = _make_scanner(context, "")
    result = scanner("x", 0)
    assert isinstance(result, tuple)
    token, end = result
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.string == "null"
    assert end == 4


# LLM-generated content at query #25
#--------------------------

def test_tokenize_json_empty_string():
    content = ""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    content = b""
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_whitespace_only():
    content = "   \n\t  "
    try:
        tokenize_json(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position == Position(column_no=1, line_no=1, char_index=0)

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

def test_tokenize_json_integer():
    content = "42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.string == "42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_float():
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

def test_tokenize_json_empty_object():
    content = "{}"
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_object():
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{"key": "value"}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=16, char_index=15)

def test_tokenize_json_empty_array():
    content = "[]"
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.string == "[]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_simple_array():
    content = "[1, 2, 3]"
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == "[1, 2, 3]"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=9, char_index=8)

def test_tokenize_json_nested_structure():
    content = '{"a": [1, {"b": true}]}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    expected = {"a": [1, {"b": True}]}
    assert token.value == expected
    assert token.string == '{"a": [1, {"b": true}]}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=24, char_index=23)

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
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.char_index > 0

def test_tokenize_json_multiline():
    content = '{\n  "key": "value"\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.string == '{\n  "key": "value"\n}'
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=3, column_no=1, char_index=20)

def test_tokenize_json_negative_number():
    content = "-42"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == -42
    assert token.string == "-42"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=3, char_index=2)

def test_tokenize_json_exponential_number():
    content = "1.23e-4"
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    assert token.string == "1.23e-4"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=7, char_index=6)


# LLM-generated content at query #26
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
    expected_key = ScalarToken("key", 1, 5, content)
    expected_value = ScalarToken("value", 9, 15, content)
    assert list(result.keys())[0] == expected_key
    assert list(result.values())[0] == expected_value
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
    expected_a = ScalarToken("a", 1, 3, content)
    expected_b = ScalarToken("b", 9, 11, content)
    assert result[expected_a].value == 1
    assert result[expected_b].value == 2
    assert end == 18

def test_TokenizingJSONObject_with_whitespace():
    content = '{ "key" : "value" }'
    memo = {}
    def scan_once(s, idx):
        start = idx
        end = idx + 7
        return ScalarToken("value", start, end - 1, content), end
    result, end = _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
    expected_key = ScalarToken("key", 2, 6, content)
    expected_value = ScalarToken("value", 11, 17, content)
    assert list(result.keys())[0] == expected_key
    assert list(result.values())[0] == expected_value
    assert end == 20

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
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting ',' delimiter" in str(e)

def test_TokenizingJSONObject_missing_quote():
    content = '{key: "value"}'
    memo = {}
    def scan_once(s, idx):
        return ScalarToken(None, idx, idx, content), idx + 1
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting property name enclosed in double quotes" in str(e)

def test_TokenizingJSONObject_stop_iteration():
    content = '{"key": }'
    memo = {}
    def scan_once(s, idx):
        raise StopIteration(idx)
    try:
        _TokenizingJSONObject((content, 0), True, scan_once, memo, content)
        assert False
    except JSONDecodeError as e:
        assert "Expecting value" in str(e)


# LLM-generated content at query #27
#--------------------------

def test_tokenize_json_empty_string():
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position == Position(column_no=1, line_no=1, char_index=0)

def test_tokenize_json_empty_bytes():
    try:
        tokenize_json(b"")
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

def test_tokenize_json_empty_object():
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.string == "{}"
    assert token.start == Position(line_no=1, column_no=1, char_index=0)
    assert token.end == Position(line_no=1, column_no=2, char_index=1)

def test_tokenize_json_object():
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

def test_tokenize_json_array():
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.string == '[1, 2, 3]'
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
        tokenize_json('{"invalid": }')
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


