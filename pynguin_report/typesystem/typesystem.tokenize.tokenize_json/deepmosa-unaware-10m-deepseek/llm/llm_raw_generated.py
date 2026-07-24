####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test float
    result = tokenize_json('12.34')
    assert isinstance(result, ScalarToken)
    assert result.value == 12.34
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    assert result.start == 0
    assert result.end == 1
    
    # Test object with properties
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    
    # Test nested object
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []
    
    # Test array with values
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    array_token = result.value[ScalarToken("array", 1, 6, '{"array": [1, 2], "nested": {"bool": true}}')]
    assert isinstance(array_token, ListToken)
    assert len(array_token.value) == 2
    
    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    
    # Test bytes input
    result = tokenize_json(b'"test"')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test unclosed string
    try:
        tokenize_json('"unclosed')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with missing quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with trailing comma
    try:
        tokenize_json('{"key": "value",}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test array with trailing comma
    try:
        tokenize_json('[1, 2,]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test correct position tracking
    content = '{\n  "key": "value"\n}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.start == 0
    assert result.end == len(content) - 1


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    
    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    
    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)
    
    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    b_value = token.value[ScalarToken("b", 12, 12, '{"a": [1, 2], "b": {"c": 3}}')]
    assert isinstance(b_value, DictToken)
    
    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test whitespace handling
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    
    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    
    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace-only string
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test numbers
    token = tokenize_json('3.14')
    assert token.value == 3.14
    
    token = tokenize_json('-42')
    assert token.value == -42
    
    token = tokenize_json('1e3')
    assert token.value == 1000.0
    
    # Test boolean false
    token = tokenize_json('false')
    assert token.value is False
    
    # Test complex nested structure
    json_str = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    users_token = token.value[ScalarToken("users", 1, 5, json_str)]
    assert isinstance(users_token, ListToken)
    assert len(users_token.value) == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    
    # Test null token
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test boolean tokens
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    
    # Test number tokens
    token = tokenize_json('42')
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test array token
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    
    # Test object token
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structure
    token = tokenize_json('{"array": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(token, DictToken)
    array_token = token.value[ScalarToken("array", 1, 6, '{"array": [1, 2], "nested": {"key": "value"}}')]
    assert isinstance(array_token, ListToken)
    assert len(array_token.value) == 2
    
    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test empty string raises ParseError
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only string raises ParseError
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON raises ParseError
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed JSON raises ParseError
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test JSON with trailing comma raises ParseError
    try:
        tokenize_json('{"key": "value",}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    json_str = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    
    # Test number with exponent
    token = tokenize_json('1.23e-4')
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    
    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    
    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    
    # Test string with escape sequences
    token = tokenize_json('"line1\\nline2"')
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2"


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_json():
    # Test basic JSON types
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    result = tokenize_json('{"nested": {"inner": [1, 2]}}')
    assert isinstance(result, DictToken)
    nested = result.value[ScalarToken("nested", 1, 7, '{"nested": {"inner": [1, 2]}}')]
    assert isinstance(nested, DictToken)
    inner = nested.value[ScalarToken("inner", 11, 16, '{"nested": {"inner": [1, 2]}}')]
    assert isinstance(inner, ListToken)
    
    # Test with bytes input
    result = tokenize_json(b'"test"')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed array
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object without quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    
    # Test numbers with different formats
    result = tokenize_json('123.456')
    assert result.value == 123.456
    
    result = tokenize_json('-42')
    assert result.value == -42
    
    result = tokenize_json('1e3')
    assert result.value == 1000.0
    
    # Test position information
    json_str = '{"a": 1, "b": 2}'
    result = tokenize_json(json_str)
    assert result.start == 0
    assert result.end == len(json_str) - 1


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    assert result.start == 0
    assert result.end == 24

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 45

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 24

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4
    assert result.start == 0
    assert result.end == 6

    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    # Test object with multiple keys
    result = tokenize_json('{"a": 1, "b": 2, "c": 3}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 3
    assert result.start == 0
    assert result.end == 22


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test list token
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test dict token
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    result = tokenize_json('{"list": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(result, DictToken)
    list_token = result.value[ScalarToken("list", 1, 5, '{"list": [1, 2], "nested": {"key": "value"}}')]
    assert isinstance(list_token, ListToken)
    assert len(list_token.value) == 2
    
    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    
    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test number parsing
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    result = tokenize_json('1e3')
    assert isinstance(result, ScalarToken)
    assert result.value == 1000.0
    
    # Test boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4
    
    # Test complex nested structure
    json_str = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    # Test null token
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test boolean tokens
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test number tokens
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test array token
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test object token
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structure
    result = tokenize_json('{"array": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    
    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string raises ParseError
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only string raises ParseError
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON raises ParseError
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string raises ParseError
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed array raises ParseError
    try:
        tokenize_json('[1, 2')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object raises ParseError
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object without quotes raises ParseError
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "active": true,
        "scores": [95.5, 87.0, 92.3],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 5
    
    # Verify all tokens have correct positions
    def verify_token_positions(token):
        assert token.start <= token.end
        if isinstance(token, DictToken):
            for k, v in token.value.items():
                verify_token_positions(k)
                verify_token_positions(v)
        elif isinstance(token, ListToken):
            for item in token.value:
                verify_token_positions(item)
    
    verify_token_positions(result)


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_value = list(result.value.values())[0]
    assert isinstance(nested_value, DictToken)
    assert list(nested_value.value.values())[0].value == 42

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with numbers
    result = tokenize_json('3.14')
    assert result.value == 3.14
    assert isinstance(result, ScalarToken)

    result = tokenize_json('-42')
    assert result.value == -42
    assert isinstance(result, ScalarToken)

    result = tokenize_json('1e10')
    assert result.value == 1e10
    assert isinstance(result, ScalarToken)

    # Test complex structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "coding"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 4
    hobbies = result.value[ScalarToken("hobbies", 0, 0, "")]
    assert isinstance(hobbies, ListToken)
    assert len(hobbies.value) == 2


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6

    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2

    token = tokenize_json('123.45')
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == 0
    assert token.end == 5

    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4

    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)
    assert token.start == 0
    assert token.end == 16

    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert token.start == 0
    assert token.end == 15

    # Test nested structures
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    assert token.start == 0
    assert token.end == 27

    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test whitespace handling
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert token.start == 2
    assert token.end == 24

    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid JSON structure
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test trailing comma (invalid in strict JSON)
    try:
        tokenize_json('[1, 2,]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test scientific notation
    token = tokenize_json('1.23e4')
    assert isinstance(token, ScalarToken)
    assert token.value == 12300.0

    # Test negative numbers
    token = tokenize_json('-42')
    assert isinstance(token, ScalarToken)
    assert token.value == -42

    # Test zero
    token = tokenize_json('0')
    assert isinstance(token, ScalarToken)
    assert token.value == 0

    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    token = tokenize_json(complex_json)
    assert isinstance(token, DictToken)
    assert len(token.value) == 4
    assert token.start == 5  # Account for leading whitespace


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.value[0].value == 1
    assert result.value[1].value == 2
    assert result.value[2].value == 3

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert isinstance(result.value[0], ListToken)
    assert isinstance(result.value[1], ListToken)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test escaped strings
    result = tokenize_json('"line1\\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test list token
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test dict token
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert isinstance(list(result.value.keys())[0], ScalarToken)
    assert isinstance(list(result.value.values())[0], ScalarToken)
    assert result.start == 0
    assert result.end == 15

    # Test nested structure
    result = tokenize_json('{"list": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert isinstance(result.value["list"], ListToken)
    assert isinstance(result.value["nested"], DictToken)

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed array
    try:
        tokenize_json('[1, 2')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test object with missing quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test object with trailing comma
    try:
        tokenize_json('{"key": "value",}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test complex nested structure
    json_str = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert len(result.value) == 4
    assert result.value["name"].value == "John"
    assert result.value["age"].value == 30
    assert len(result.value["hobbies"].value) == 2
    assert result.value["address"].value["city"].value == "Boston"


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    assert token.content == '"hello"'

    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2

    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False

    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test numbers with decimals and exponents
    token = tokenize_json('123.456')
    assert token.value == 123.456

    token = tokenize_json('1.23e4')
    assert token.value == 12300.0

    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]

    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)

    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"

    # Test nested structures
    token = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(token, DictToken)
    nested_token = list(token.value.values())[0]
    assert isinstance(nested_token, DictToken)
    inner_token = list(nested_token.value.values())[0]
    assert inner_token.value == 42

    # Test with whitespace
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1

    # Test empty structures
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0

    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0

    # Test bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test that content is preserved in tokens
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert token.content == content
    name_token = list(token.value.keys())[0]
    assert name_token.content == content


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    inner_token = list(nested_token.value.values())[0]
    assert inner_token.value == 42

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test list token
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test dict token
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structure
    result = tokenize_json('{"list": [1, 2], "nested": {"key": "value"}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    list_token = result.value[ScalarToken("list", 1, 5, '{"list": [1, 2], "nested": {"key": "value"}}')]
    nested_token = result.value[ScalarToken("nested", 16, 22, '{"list": [1, 2], "nested": {"key": "value"}}')]
    assert isinstance(list_token, ListToken)
    assert isinstance(nested_token, DictToken)

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed JSON
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed string
    try:
        tokenize_json('"unclosed')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test complex nested structure
    json_str = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}], "count": 2}'
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    users_token = list(result.value.values())[0]
    count_token = list(result.value.values())[1]
    assert isinstance(users_token, ListToken)
    assert isinstance(count_token, ScalarToken)
    assert count_token.value == 2
    assert len(users_token.value) == 2
    assert all(isinstance(u, DictToken) for u in users_token.value)

    # Test with escaped characters in strings
    result = tokenize_json(r'"line1\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"

    # Test with unicode
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"

    # Test with scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json():
    # Test basic JSON types
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": [1, 2]}}')
    assert isinstance(result, DictToken)
    nested = result.value[list(result.value.keys())[0]]
    assert isinstance(nested, DictToken)
    inner = nested.value[list(nested.value.keys())[0]]
    assert isinstance(inner, ListToken)
    assert len(inner.value) == 2

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed JSON string
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed array
    try:
        tokenize_json('[1, 2')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test complex nested structure
    complex_json = '''
    {
        "name": "test",
        "values": [1, 2, 3],
        "nested": {
            "flag": true,
            "count": null
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 3


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    
    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test nested structures
    result = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["a"], ListToken)
    assert isinstance(result.value["b"], DictToken)
    
    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test unclosed array
    try:
        tokenize_json('[1, 2')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test number parsing
    result = tokenize_json('3.14')
    assert result.value == 3.14
    
    result = tokenize_json('-42')
    assert result.value == -42
    
    result = tokenize_json('1e3')
    assert result.value == 1000.0
    
    # Test boolean values
    result = tokenize_json('false')
    assert result.value is False
    
    # Test complex nested structure
    json_str = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert result.value["name"].value == "John"
    assert result.value["age"].value == 30
    assert len(result.value["hobbies"].value) == 2
    assert result.value["address"].value["city"].value == "Boston"


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_json():
    # Test basic JSON parsing
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}
    
    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    inner_dict = result.value["outer"]
    assert isinstance(inner_dict, DictToken)
    assert inner_dict.value == {"inner": ScalarToken(42, 18, 19, '{"outer": {"inner": 42}}')}
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert result.value[0] == ScalarToken(1, 1, 1, '[1, 2, 3]')
    
    # Test strings
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test numbers
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test booleans
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"test": "bytes"}')
    assert isinstance(result, DictToken)
    assert result.value == {"test": ScalarToken("bytes", 9, 15, '{"test": "bytes"}')}
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []
    
    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert result.value == {"key": ScalarToken("value", 16, 22, '  {  "key"  :  "value"  }  ')}
    
    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "test",
        "numbers": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "name" in result.value
    assert "numbers" in result.value
    assert "nested" in result.value
    assert isinstance(result.value["numbers"], ListToken)
    assert isinstance(result.value["nested"], DictToken)


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2
    
    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test array token
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8
    
    # Test nested array
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 16
    
    # Test object token
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key = list(result.value.keys())[0]
    value = result.value[key]
    assert isinstance(key, ScalarToken)
    assert key.value == "key"
    assert isinstance(value, ScalarToken)
    assert value.value == "value"
    assert result.start == 0
    assert result.end == 15
    
    # Test nested object
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    assert result.start == 0
    assert result.end == 24
    
    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 47
    
    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object without quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test with escaped characters
    result = tokenize_json('"line1\\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"
    
    # Test unicode in string
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"
    
    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test zero
    result = tokenize_json('0')
    assert isinstance(result, ScalarToken)
    assert result.value == 0
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    
    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    
    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested = result.value["nested"]
    assert isinstance(nested, DictToken)
    assert nested.value["inner"].value == 42
    
    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    
    # Test bytes input
    result = tokenize_json(b'"test"')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    result = tokenize_json('-123.456e-10')
    assert isinstance(result, ScalarToken)
    assert isinstance(result.value, float)
    
    # Test scientific notation
    result = tokenize_json('1.23e+5')
    assert isinstance(result, ScalarToken)
    assert result.value == 123000.0
    
    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test unclosed string
    try:
        tokenize_json('"unclosed')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test invalid JSON structure
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test with escaped characters in strings
    result = tokenize_json(r'"line1\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"
    
    # Test unicode in strings
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"
    
    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        },
        "number": 42
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert isinstance(result.value["array"], ListToken)
    assert isinstance(result.value["nested"], DictToken)
    assert result.value["nested"].value["bool"].value is True
    assert result.value["nested"].value["null"].value is None
    assert result.value["number"].value == 42


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    
    # Test numbers
    token = tokenize_json('42')
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1
    
    # Test floats
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3
    
    # Test booleans
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3
    
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4
    
    # Test null
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3
    
    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.start == 0
    assert token.end == 1
    
    # Test object with properties
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    
    # Test nested object
    token = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(token, DictToken)
    outer_value = list(token.value.values())[0]
    assert isinstance(outer_value, DictToken)
    inner_value = list(outer_value.value.values())[0]
    assert isinstance(inner_value, ScalarToken)
    assert inner_value.value == 42
    
    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.start == 0
    assert token.end == 1
    
    # Test array with elements
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    for i, item in enumerate(token.value):
        assert isinstance(item, ScalarToken)
        assert item.value == i + 1
    
    # Test complex nested structure
    token = tokenize_json('{"array": [1, {"nested": true}], "string": "test"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    
    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test empty string raises ParseError
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only string raises ParseError
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON raises ParseError
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with unquoted keys raises error
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object missing comma
    try:
        tokenize_json('{"a": 1 "b": 2}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test array missing comma
    try:
        tokenize_json('[1 2]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #21
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    assert result.start == 0
    assert result.end == 24

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 45

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    result = tokenize_json(b'[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with escaped characters in strings
    result = tokenize_json('"line1\\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"

    result = tokenize_json('"quoted \\"string\\""')
    assert isinstance(result, ScalarToken)
    assert result.value == 'quoted "string"'

    # Test numbers with exponents
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    result = tokenize_json('-1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == -0.000123

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test array with trailing whitespace
    result = tokenize_json('[1, 2, 3]   ')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert result.end == 8  # Should end at ']', not including trailing whitespace


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    
    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    
    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)
    
    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    assert "a" in [k.value for k in token.value.keys()]
    assert "b" in [k.value for k in token.value.keys()]
    
    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test with floating point numbers
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test with scientific notation
    token = tokenize_json('1.23e-4')
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    token = tokenize_json(complex_json)
    assert isinstance(token, DictToken)
    assert len(token.value) == 4
    assert "name" in [k.value for k in token.value.keys()]
    assert "age" in [k.value for k in token.value.keys()]
    
    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    
    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    
    # Test object with spaces
    token = tokenize_json('{ "key" : "value" }')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    assert len(nested_token.value) == 1
    assert result.start == 0
    assert result.end == 25

    # Test with whitespace
    result = tokenize_json('  { "key" : 123 }  ')
    assert isinstance(result, DictToken)
    assert result.start == 2
    assert result.end == 16

    # Test bytes input
    result = tokenize_json(b'{"bytes": "input"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('{"unclosed": "string}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    assert result.start == 0
    assert result.end == 5

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    assert result.start == 0
    assert result.end == 2

    # Test complex structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null
        },
        "string": "test"
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 3
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]
    assert "string" in [k.value for k in result.value.keys()]


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    
    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    
    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    result = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    
    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test correct position tracking
    result = tokenize_json('  "test"  ')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    assert result.start == 2
    assert result.end == 7
    
    # Test number with decimal
    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    
    # Test number with exponent
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    
    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 4
    
    # Test array with mixed types
    result = tokenize_json('[1, "two", true, null]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 4
    assert result.value[0].value == 1
    assert result.value[1].value == "two"
    assert result.value[2].value is True
    assert result.value[3].value is None


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    inner_token = list(nested_token.value.values())[0]
    assert isinstance(inner_token, ScalarToken)
    assert inner_token.value == 42

    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed array
    try:
        tokenize_json('[1, 2, ]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test complex structure
    complex_json = '''
    {
        "string": "value",
        "number": 123.45,
        "boolean": true,
        "null": null,
        "array": [1, 2, 3],
        "object": {"nested": "value"}
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 6
    
    # Verify all tokens have correct positions
    def verify_token_positions(token):
        assert token.start <= token.end
        if isinstance(token, DictToken):
            for k, v in token.value.items():
                verify_token_positions(k)
                verify_token_positions(v)
        elif isinstance(token, ListToken):
            for item in token.value:
                verify_token_positions(item)
    
    verify_token_positions(result)


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    assert result.start == 0
    assert result.end == 25

    # Test with whitespace
    result = tokenize_json('  { "key" : "value" }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 19

    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed JSON (missing closing brace)
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed JSON (unquoted key)
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4
    assert result.start == 0
    assert result.end == 6

    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6

    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2

    token = tokenize_json('123.45')
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == 0
    assert token.end == 5

    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4

    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    assert token.start == 0
    assert token.end == 8

    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)
    assert token.start == 0
    assert token.end == 16

    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert "key" in token.value
    assert isinstance(token.value["key"], ScalarToken)
    assert token.value["key"].value == "value"
    assert token.start == 0
    assert token.end == 15

    # Test nested objects
    token = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(token, DictToken)
    assert "outer" in token.value
    assert isinstance(token.value["outer"], DictToken)
    inner = token.value["outer"]
    assert "inner" in inner.value
    assert inner.value["inner"].value == 42
    assert token.start == 0
    assert token.end == 24

    # Test complex structure
    token = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(token, DictToken)
    assert "array" in token.value
    assert "nested" in token.value
    assert isinstance(token.value["array"], ListToken)
    assert isinstance(token.value["nested"], DictToken)
    assert token.value["nested"].value["bool"].value is True

    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed array
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test object with unquoted keys
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with escaped characters in strings
    token = tokenize_json('"line1\\nline2"')
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2"

    # Test with unicode characters
    token = tokenize_json('"café"')
    assert isinstance(token, ScalarToken)
    assert token.value == "café"

    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test number with exponent
    token = tokenize_json('1.23e-4')
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    assert result.start == 0
    assert result.end == 25

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 24

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    assert result.start == 0
    assert result.end == 5

    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    inner_token = list(nested_token.value.values())[0]
    assert inner_token.value == 42

    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test position information in tokens
    result = tokenize_json('{"a": 1, "b": [2, 3]}')
    assert isinstance(result, DictToken)
    assert result.start == 0
    assert result.end == len('{"a": 1, "b": [2, 3]}') - 1
    
    # Verify array token positions
    b_token = result.value[ScalarToken("b", 7, 9, '{"a": 1, "b": [2, 3]}')]
    assert isinstance(b_token, ListToken)
    assert b_token.start == 12
    assert b_token.end == 18

    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex structure
    complex_json = '''
    {
        "name": "test",
        "values": [1, 2, 3],
        "nested": {
            "flag": true,
            "text": null
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 3


# LLM-generated content at query #8
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_value = list(result.value.values())[0]
    assert isinstance(nested_value, DictToken)
    inner_value = list(nested_value.value.values())[0]
    assert isinstance(inner_value, ScalarToken)
    assert inner_value.value == 42

    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 24

    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    array_token = result.value[ScalarToken("array", 13, 17, complex_json)]
    assert isinstance(array_token, ListToken)
    assert len(array_token.value) == 3


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6

    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2

    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]

    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)

    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested objects
    token = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(token, DictToken)
    outer_value = list(token.value.values())[0]
    assert isinstance(outer_value, DictToken)

    # Test bytes input
    token = tokenize_json(b'"test"')
    assert isinstance(token, ScalarToken)
    assert token.value == "test"

    # Test whitespace handling
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1

    # Test empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 0

    # Test empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 0

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed array
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test object without quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test float numbers
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test scientific notation
    token = tokenize_json('1.23e4')
    assert isinstance(token, ScalarToken)
    assert token.value == 12300.0

    # Test negative numbers
    token = tokenize_json('-42')
    assert isinstance(token, ScalarToken)
    assert token.value == -42

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    token = tokenize_json(complex_json)
    assert isinstance(token, DictToken)
    assert "array" in token.value
    assert "nested" in token.value
    array_token = token.value[ScalarToken("array", 0, 0, "")]
    nested_token = token.value[ScalarToken("nested", 0, 0, "")]
    assert isinstance(array_token, ListToken)
    assert isinstance(nested_token, DictToken)


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.value[0].value == 1
    assert result.value[1].value == 2
    assert result.value[2].value == 3
    assert result.start == 0
    assert result.end == 8
    
    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 16
    
    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15
    
    # Test nested structures
    result = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 27
    
    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 24
    
    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1
    
    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3
    
    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    assert result.start == 0
    assert result.end == 5
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    assert result.start == 0
    assert result.end == 2
    
    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "test",
        "values": [1, 2, 3],
        "nested": {
            "inner": true,
            "items": []
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 3
    assert result.start == 5  # Account for leading whitespace


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "key" in result.value
    assert isinstance(result.value["key"], ScalarToken)
    assert result.value["key"].value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = result.value["outer"]
    assert isinstance(outer_value, DictToken)
    assert outer_value.value["inner"].value == 42

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["array"], ListToken)
    assert isinstance(result.value["nested"], DictToken)
    assert result.value["nested"].value["bool"].value is True

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    result = tokenize_json(b'[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed array
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test object with unquoted keys
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test array with whitespace
    result = tokenize_json('[ 1 , 2 , 3 ]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert result.value[0].value == 1
    assert result.value[1].value == 2
    assert result.value[2].value == 3

    # Test object with whitespace
    result = tokenize_json('{ "key" : "value" }')
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "value"

    # Test escaped characters in strings
    result = tokenize_json('"line1\\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"

    # Test unicode in strings
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"

    # Test bytes with unicode
    result = tokenize_json(b'"caf\xc3\xa9"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6

    token = tokenize_json("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1

    token = tokenize_json("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3

    token = tokenize_json("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3

    token = tokenize_json("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4

    token = tokenize_json("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3

    # Test arrays
    token = tokenize_json('[1, "two", true]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert isinstance(token.value[0], ScalarToken)
    assert token.value[0].value == 1
    assert isinstance(token.value[1], ScalarToken)
    assert token.value[1].value == "two"
    assert isinstance(token.value[2], ScalarToken)
    assert token.value[2].value is True
    assert token.start == 0
    assert token.end == 15

    # Test objects
    token = tokenize_json('{"key": "value", "num": 42}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    keys = list(token.value.keys())
    values = list(token.value.values())
    assert isinstance(keys[0], ScalarToken)
    assert keys[0].value == "key"
    assert isinstance(values[0], ScalarToken)
    assert values[0].value == "value"
    assert isinstance(keys[1], ScalarToken)
    assert keys[1].value == "num"
    assert isinstance(values[1], ScalarToken)
    assert values[1].value == 42
    assert token.start == 0
    assert token.end == 25

    # Test nested structures
    token = tokenize_json('{"arr": [1, 2], "obj": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    arr_token = token.value[list(token.value.keys())[0]]
    obj_token = token.value[list(token.value.keys())[1]]
    assert isinstance(arr_token, ListToken)
    assert isinstance(obj_token, DictToken)
    assert len(arr_token.value) == 2
    assert len(obj_token.value) == 1

    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test empty string
    try:
        tokenize_json("")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace only
    try:
        tokenize_json("   \n\t  ")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{"key": value}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json("[1, 2, 3")
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with escaped characters
    token = tokenize_json('"line1\\nline2"')
    assert isinstance(token, ScalarToken)
    assert token.value == "line1\nline2"

    # Test with unicode
    token = tokenize_json('"café"')
    assert isinstance(token, ScalarToken)
    assert token.value == "café"

    # Test empty object
    token = tokenize_json("{}")
    assert isinstance(token, DictToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test empty array
    token = tokenize_json("[]")
    assert isinstance(token, ListToken)
    assert len(token.value) == 0
    assert token.start == 0
    assert token.end == 1

    # Test object with whitespace
    token = tokenize_json('{ "key" : "value" }')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    key = list(token.value.keys())[0]
    value = token.value[key]
    assert key.value == "key"
    assert value.value == "value"


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested structures
    result = tokenize_json('{"nested": {"inner": 42}}')
    assert isinstance(result, DictToken)
    nested_token = list(result.value.values())[0]
    assert isinstance(nested_token, DictToken)
    inner_token = list(nested_token.value.values())[0]
    assert inner_token.value == 42

    # Test with whitespace
    result = tokenize_json('  { "key" : 123 }  ')
    assert isinstance(result, DictToken)
    assert result.start == 2
    assert result.end == 16

    # Test bytes input
    result = tokenize_json(b'"test"')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test malformed string
    try:
        tokenize_json('{"unclosed": ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed array
    try:
        tokenize_json('[1, 2, 3')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "hello"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert result.value[0].value == 1
    assert result.value[1].value == 2
    assert result.value[2].value == 3
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 16

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    assert result.start == 0
    assert result.end == 24

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 46

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
        assert e.position.line_no == 1
        assert e.position.column_no == 1

    # Test whitespace only
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON
    try:
        tokenize_json('{"unclosed":')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting value" in e.text or "Expecting property name" in e.text

    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid number
    try:
        tokenize_json('12.34.56')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with escaped characters in strings
    result = tokenize_json(r'"line1\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"

    # Test with unicode
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test object with whitespace
    result = tokenize_json('{ "key" : "value" }')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 0
    assert result.end == 19


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 15

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested structures
    result = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 26

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 24

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test floating point numbers
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    assert result.start == 0
    assert result.end == 5

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    assert result.start == 0
    assert result.end == 2

    # Test false value
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 4
    assert result.start == 5  # Account for leading whitespace


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"

    # Test nested structures
    result = tokenize_json('{"a": [1, 2], "b": {"c": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert "a" in [k.value for k in result.value.keys()]
    assert "b" in [k.value for k in result.value.keys()]

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    assert key_token.value == "key"

    # Test number formats
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    # Test complex nested structure
    complex_json = '''
    {
        "array": [1, 2, 3],
        "nested": {
            "bool": true,
            "null": null,
            "string": "test"
        }
    }
    '''
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('123.45')
    assert isinstance(result, ScalarToken)
    assert result.value == 123.45
    assert result.start == 0
    assert result.end == 5

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test arrays
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert all(isinstance(t, ScalarToken) for t in result.value)
    assert [t.value for t in result.value] == [1, 2, 3]
    assert result.start == 0
    assert result.end == 8

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert all(isinstance(t, ListToken) for t in result.value)
    assert result.start == 0
    assert result.end == 16

    # Test objects
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    assert result.start == 0
    assert result.end == 15

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    assert result.start == 0
    assert result.end == 24

    # Test complex structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2
    assert result.start == 0
    assert result.end == 45

    # Test with bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 0
    assert result.start == 0
    assert result.end == 1

    # Test whitespace handling
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert result.start == 2
    assert result.end == 23

    # Test error cases
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    try:
        tokenize_json('   ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test unclosed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid JSON structure
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    
    # Test numbers
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start == 0
    assert result.end == 1
    
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start == 0
    assert result.end == 3
    
    # Test booleans
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start == 0
    assert result.end == 4
    
    # Test null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    assert result.start == 0
    assert result.end == 1
    
    # Test simple object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"
    
    # Test nested object
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_value = list(result.value.values())[0]
    assert isinstance(outer_value, DictToken)
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []
    assert result.start == 0
    assert result.end == 1
    
    # Test array with values
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    for i, token in enumerate(result.value):
        assert isinstance(token, ScalarToken)
        assert token.value == i + 1
    
    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2], "nested": {"bool": true}}')
    assert isinstance(result, DictToken)
    assert "array" in [k.value for k in result.value.keys()]
    assert "nested" in [k.value for k in result.value.keys()]
    
    # Test bytes input
    result = tokenize_json(b'"test"')
    assert isinstance(result, ScalarToken)
    assert result.value == "test"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n\t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": }')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with missing quotes
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with trailing comma (invalid in strict JSON)
    try:
        tokenize_json('{"key": "value",}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test array with trailing comma (invalid in strict JSON)
    try:
        tokenize_json('[1, 2, 3,]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar tokens
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start == 0
    assert result.end == 6
    assert result.content == '"hello"'

    result = tokenize_json('123')
    assert isinstance(result, ScalarToken)
    assert result.value == 123
    assert result.start == 0
    assert result.end == 2

    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start == 0
    assert result.end == 3

    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start == 0
    assert result.end == 3

    # Test float
    result = tokenize_json('12.34')
    assert isinstance(result, ScalarToken)
    assert result.value == 12.34
    assert result.start == 0
    assert result.end == 4

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    assert result.start == 0
    assert result.end == 2

    # Test scientific notation
    result = tokenize_json('1.23e4')
    assert isinstance(result, ScalarToken)
    assert result.value == 12300.0
    assert result.start == 0
    assert result.end == 5

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    assert result.start == 0
    assert result.end == 1

    # Test simple object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    value_token = result.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert key_token.value == "key"
    assert isinstance(value_token, ScalarToken)
    assert value_token.value == "value"

    # Test nested object
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    outer_dict = result.value
    assert len(outer_dict) == 1
    outer_key = list(outer_dict.keys())[0]
    assert outer_key.value == "outer"
    inner_token = outer_dict[outer_key]
    assert isinstance(inner_token, DictToken)
    inner_dict = inner_token.value
    assert len(inner_dict) == 1
    inner_key = list(inner_dict.keys())[0]
    assert inner_key.value == "inner"
    assert inner_dict[inner_key].value == 42

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []
    assert result.start == 0
    assert result.end == 1

    # Test array with values
    result = tokenize_json('[1, "two", true]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 3
    assert result.value[0].value == 1
    assert result.value[1].value == "two"
    assert result.value[2].value is True

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, 3], "nested": {"bool": false}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    key_token = list(result.value.keys())[0]
    assert key_token.value == "key"
    assert result.value[key_token].value == "value"

    # Test bytes input
    result = tokenize_json(b'"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.char_index == 0

    # Test whitespace-only string
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"

    # Test invalid JSON - missing closing quote
    try:
        tokenize_json('"hello')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid JSON - malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid JSON - unexpected token
    try:
        tokenize_json('{key: "value"}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test invalid JSON - missing comma in array
    try:
        tokenize_json('[1 2]')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with escaped characters in string
    result = tokenize_json('"line1\\nline2"')
    assert isinstance(result, ScalarToken)
    assert result.value == "line1\nline2"

    # Test with unicode characters
    result = tokenize_json('"café"')
    assert isinstance(result, ScalarToken)
    assert result.value == "café"

    # Test object with multiple keys
    result = tokenize_json('{"a": 1, "b": 2, "c": 3}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 3
    values = [token.value for token in result.value.values()]
    assert set(values) == {1, 2, 3}


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_json():
    # Test basic scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 6
    
    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2
    
    token = tokenize_json('123.45')
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == 0
    assert token.end == 5
    
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3
    
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == 4
    
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3
    
    # Test arrays
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert token.start == 0
    assert token.end == 8
    assert all(isinstance(t, ScalarToken) for t in token.value)
    assert [t.value for t in token.value] == [1, 2, 3]
    
    # Test nested arrays
    token = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(t, ListToken) for t in token.value)
    
    # Test objects
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert token.start == 0
    assert token.end == 15
    key_token = list(token.value.keys())[0]
    value_token = token.value[key_token]
    assert isinstance(key_token, ScalarToken)
    assert isinstance(value_token, ScalarToken)
    assert key_token.value == "key"
    assert value_token.value == "value"
    
    # Test nested structures
    token = tokenize_json('{"a": [1, 2], "b": {"c": 3}}')
    assert isinstance(token, DictToken)
    assert len(token.value) == 2
    a_value = token.value[ScalarToken("a", 1, 3, '{"a": [1, 2], "b": {"c": 3}}')]
    b_value = token.value[ScalarToken("b", 13, 15, '{"a": [1, 2], "b": {"c": 3}}')]
    assert isinstance(a_value, ListToken)
    assert isinstance(b_value, DictToken)
    
    # Test with bytes input
    token = tokenize_json(b'"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    # Test empty string
    try:
        tokenize_json('')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test whitespace only
    try:
        tokenize_json('   \n  \t  ')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test invalid JSON
    try:
        tokenize_json('{invalid}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting property name" in e.text
    
    # Test malformed string
    try:
        tokenize_json('"unclosed string')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed array
    try:
        tokenize_json('[1, 2,')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test malformed object
    try:
        tokenize_json('{"key": "value"')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test object with trailing comma (invalid in strict JSON)
    try:
        tokenize_json('{"key": "value",}')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '''
    {
        "name": "John",
        "age": 30,
        "hobbies": ["reading", "swimming"],
        "address": {
            "street": "123 Main St",
            "city": "Boston"
        }
    }
    '''
    token = tokenize_json(complex_json)
    assert isinstance(token, DictToken)
    assert len(token.value) == 4
    
    # Test that positions are correctly calculated
    simple = '{"a": 1}'
    token = tokenize_json(simple)
    key = ScalarToken("a", 1, 3, simple)
    assert key in token.value
    value = token.value[key]
    assert value.value == 1
    assert value.start == 6
    assert value.end == 6


