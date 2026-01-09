####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test with valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15

    # Test with bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    print("All tests passed!")



# LLM-generated content at query #2
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 2 passed")

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    print("Test case 3 passed")

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    print("Test case 4 passed")

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    print("Test case 5 passed")

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    print("Test case 6 passed")

    # Test case 7: Empty JSON object
    content = '{}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    print("Test case 7 passed")

    # Test case 8: Empty JSON array
    content = '[]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    print("Test case 8 passed")

    # Test case 9: Nested JSON object
    content = '{"person": {"name": "Alice", "age": 25}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "Alice", "age": 25}}
    print("Test case 9 passed")

    # Test case 10: JSON with whitespace
    content = '  {  "key"  :  "value"  }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 10 passed")

    # Test case 11: Invalid JSON (missing closing brace)
    content = '{"name": "John"'
    try:
        token = tokenize_json(content)
        print("Test case 11 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "parse_error"
        print("Test case 11 passed")

    # Test case 12: Invalid JSON (trailing comma)
    content = '{"name": "John",}'
    try:
        token = tokenize_json(content)
        print("Test case 12 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "parse_error"
        print("Test case 12 passed")

    # Test case 13: Empty string
    content = ''
    try:
        token = tokenize_json(content)
        print("Test case 13 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "no_content"
        print("Test case 13 passed")

    # Test case 14: Bytes input
    content = b'{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 14 passed")

    # Test case 15: Complex nested structure
    content = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    expected = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ],
        "total": 2
    }
    assert token.value == expected
    print("Test case 15 passed")

    print("All test cases passed!")

# Run the unit tests
if __name__ == "__main__":
    test_tokenize_json()


# LLM-generated content at query #3
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == len(content)

    # Test with JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with JSON true
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with JSON false
    content = 'false'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is False
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test with JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    assert token.start == 0
    assert token.end == len(content) - 1

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #4
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ''
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 24
        assert e.position.line_no == 1
        assert e.position.char_index == 23
    
    # Test with JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    
    # Test with JSON string containing special characters
    content = '{"name": "John\\"Doe"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": 'John"Doe'}
    
    # Test with JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test with JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test with JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None



# LLM-generated content at query #5
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test with invalid JSON
    try:
        tokenize_json("{")
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #6
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    print("All tests passed.")



# LLM-generated content at query #7
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 3: Empty string
    content = ""
    try:
        tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 5: JSON with nested objects
    content = '{"person": {"name": "John", "age": 30}}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"person": {"name": "John", "age": 30}}
    
    # Test case 6: JSON with array
    content = '[1, 2, 3]'
    result = tokenize_json(content)
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test case 7: JSON with boolean and null values
    content = '{"active": true, "value": null}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"active": True, "value": None}
    
    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #8
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    
    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    
    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test case 7: Empty string (should raise ParseError)
    content = ''
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
    
    # Test case 8: Invalid JSON (should raise ParseError)
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
    
    # Test case 9: Bytes input
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #9
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content

    # Test case 7: Empty string
    content = ''
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test case 8: Invalid JSON
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 24
        assert e.position.line_no == 1
        assert e.position.char_index == 23

    # Test case 9: Bytes input
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1
    assert token.content == content.decode("utf-8")

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #10
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 7: Empty string
    content = ''
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test case 8: Invalid JSON
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 25
        assert e.position.line_no == 1
        assert e.position.char_index == 24

    # Test case 9: Bytes input
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 10: Nested JSON object
    content = '{"person": {"name": "John", "age": 30}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value["person"], DictToken)
    assert token.value["person"].value == {"name": "John", "age": 30}

    # Test case 11: JSON array with nested objects
    content = '[{"name": "John"}, {"name": "Jane"}]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(item, DictToken) for item in token.value)
    assert token.value[0].value == {"name": "John"}
    assert token.value[1].value == {"name": "Jane"}

    # Test case 12: JSON with whitespace
    content = '  { "name" : "John" }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John"}
    assert token.start == 2
    assert token.end == len(content) - 3

    # Test case 13: JSON with escaped characters
    content = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}

    # Test case 14: JSON with unicode characters
    content = '{"name": "Jöhn"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "Jöhn"}

    # Test case 15: JSON with scientific notation
    content = '{"value": 1.23e-4}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"value": 0.000123}

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #11
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError"

    # Test with invalid JSON
    try:
        tokenize_json("{")
    except ParseError as e:
        assert e.text == "Expecting value."
        assert e.code == "parse_error"
        assert e.position.column_no == 2
        assert e.position.line_no == 1
        assert e.position.char_index == 1
    else:
        assert False, "Expected ParseError"

    # Test with valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}
    assert token.start == 0
    assert token.end == 15

    # Test with bytes
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}
    assert token.start == 0
    assert token.end == 15

    # Test with number
    token = tokenize_json('{"key": 123}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(123, 7, 9, '{"key": 123}')}
    assert token.start == 0
    assert token.end == 11

    # Test with boolean
    token = tokenize_json('{"key": true}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(True, 7, 10, '{"key": true}')}
    assert token.start == 0
    assert token.end == 12

    # Test with null
    token = tokenize_json('{"key": null}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(None, 7, 10, '{"key": null}')}
    assert token.start == 0
    assert token.end == 12

    # Test with array
    token = tokenize_json('{"key": [1, 2, 3]}')
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], ListToken)
    assert token.value["key"].value == [
        ScalarToken(1, 9, 9, '{"key": [1, 2, 3]}'),
        ScalarToken(2, 12, 12, '{"key": [1, 2, 3]}'),
        ScalarToken(3, 15, 15, '{"key": [1, 2, 3]}'),
    ]
    assert token.start == 0
    assert token.end == 18

    # Test with nested object
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], DictToken)
    assert token.value["key"].value == {
        "nested": ScalarToken("value", 18, 24, '{"key": {"nested": "value"}}')
    }
    assert token.start == 0
    assert token.end == 28

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #12
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    json_str = '{"name": "John", "age": 30}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON array
    json_str = '[1, 2, 3]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 2 passed")

    # Test case 3: Valid JSON string
    json_str = '"Hello, World!"'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    print("Test case 3 passed")

    # Test case 4: Valid JSON number
    json_str = '42'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    print("Test case 4 passed")

    # Test case 5: Valid JSON boolean
    json_str = 'true'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    print("Test case 5 passed")

    # Test case 6: Valid JSON null
    json_str = 'null'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    print("Test case 6 passed")

    # Test case 7: Empty JSON object
    json_str = '{}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {}
    print("Test case 7 passed")

    # Test case 8: Empty JSON array
    json_str = '[]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == []
    print("Test case 8 passed")

    # Test case 9: Nested JSON object
    json_str = '{"person": {"name": "Alice", "age": 25}}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "Alice", "age": 25}}
    print("Test case 9 passed")

    # Test case 10: Nested JSON array
    json_str = '[[1, 2], [3, 4]]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    print("Test case 10 passed")

    # Test case 11: JSON with whitespace
    json_str = '  { "key" : "value" }  '
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 11 passed")

    # Test case 12: JSON with escaped characters
    json_str = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}
    print("Test case 12 passed")

    # Test case 13: JSON with unicode characters
    json_str = '{"emoji": "😀"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"emoji": "😀"}
    print("Test case 13 passed")

    # Test case 14: JSON with scientific notation
    json_str = '{"number": 1.23e-4}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"number": 0.000123}
    print("Test case 14 passed")

    # Test case 15: JSON with negative numbers
    json_str = '{"temperature": -10}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"temperature": -10}
    print("Test case 15 passed")

    # Test case 16: JSON with zero
    json_str = '{"zero": 0}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"zero": 0}
    print("Test case 16 passed")

    # Test case 17: JSON with empty string
    json_str = '{"empty": ""}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"empty": ""}
    print("Test case 17 passed")

    # Test case 18: JSON with special characters in keys
    json_str = '{"key-with-dash": "value"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key-with-dash": "value"}
    print("Test case 18 passed")

    # Test case 19: JSON with multiple nested levels
    json_str = '{"a": {"b": {"c": {"d": "value"}}}}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": {"c": {"d": "value"}}}}
    print("Test case 19 passed")

    # Test case 20: JSON with mixed types in array
    json_str = '[1, "two", true, null]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", True, None]
    print("Test case 20 passed")

    # Test case 21: JSON with trailing comma (should fail)
    json_str = '{"a": 1,}'
    try:
        token = tokenize_json(json_str)
        print("Test case 21 failed: Expected ParseError")
    except ParseError:
        print("Test case 21 passed")

    # Test case 22: JSON with missing closing brace (should fail)
    json_str = '{"a": 1'
    try:
        token = tokenize_json(json_str)
        print("Test case 22 failed: Expected ParseError")
    except ParseError:
        print("Test case 22 passed")

    # Test case 23: JSON with invalid escape sequence (should fail)
    json_str = '{"invalid": "\\x"}'
    try:
        token = tokenize_json(json_str)
        print("Test case 23 failed: Expected ParseError")
    except ParseError:
        print("Test case 23 passed")

    # Test case 24: JSON with invalid number (should fail)
    json_str = '{"number": 123.}'
    try:
        token = tokenize_json(json_str)
        print("Test case 24 failed: Expected ParseError")
    except ParseError:
        print("Test case 24 passed")

    # Test case 25: JSON with duplicate keys (should parse but behavior may vary)
    json_str = '{"key": "first", "key": "second"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    # Note: The JSON specification does not define behavior for duplicate keys.
    # The tokenizer may keep the last occurrence.
    print("Test case 25 passed")

    # Test case 26: JSON with large integer
    json_str = '{"big": 12345678901234567890}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"big": 12345678901234567890}
    print("Test case 26 passed")

    # Test case 27: JSON with float
    json_str = '{"pi": 3.14159}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"pi": 3.14159}
    print("Test case 27 passed")

    # Test case 28: JSON with exponent without fraction
    json_str = '{"large": 1e6}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"large": 1000000.0}
    print("Test case 28 passed")

    # Test case 29: JSON with negative exponent
    json_str = '{"small": 1e-6}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"small": 1e-06}
    print("Test case 29 passed")

    # Test case 30: JSON with positive exponent
    json_str = '{"positive": 1e+6}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"positive": 1000000.0}
    print("Test case 30 passed")

    # Test case 31: JSON with all whitespace characters
    json_str = '\n\t\r {"key":\n\t\r "value"\n\t\r }\n\t\r '
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 31


# LLM-generated content at query #13
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.position.column_no == 27
        assert e.position.line_no == 1
        assert e.position.char_index == 26
    
    print("All tests passed!")



# LLM-generated content at query #14
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError"

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, '[1, 2, 3]'), ScalarToken(2, 4, 4, '[1, 2, 3]'), ScalarToken(3, 7, 7, '[1, 2, 3]')]

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15
    else:
        assert False, "Expected ParseError"

    # Test case 5: Bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 6: Whitespace only
    try:
        tokenize_json('   ')
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError"

    # Test case 7: JSON with nested structures
    token = tokenize_json('{"key": [1, 2, {"nested": "value"}]}')
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], ListToken)
    assert token.value["key"].value[2].value == {"nested": ScalarToken("value", 30, 36, '{"key": [1, 2, {"nested": "value"}]}')}

    # Test case 8: JSON with special characters
    token = tokenize_json('{"key": "value with \\"quotes\\""}')
    assert isinstance(token, DictToken)
    assert token.value["key"].value == 'value with "quotes"'

    # Test case 9: JSON with numbers
    token = tokenize_json('{"int": 42, "float": 3.14, "negative": -10}')
    assert isinstance(token, DictToken)
    assert token.value["int"].value == 42
    assert token.value["float"].value == 3.14
    assert token.value["negative"].value == -10

    # Test case 10: JSON with boolean and null
    token = tokenize_json('{"true": true, "false": false, "null": null}')
    assert isinstance(token, DictToken)
    assert token.value["true"].value is True
    assert token.value["false"].value is False
    assert token.value["null"].value is None

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #15
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 3: Empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 24
        assert e.position.line_no == 1
        assert e.position.char_index == 23
    
    # Test case 5: JSON with nested objects
    content = '{"person": {"name": "John", "age": 30}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "John", "age": 30}}
    
    # Test case 6: JSON with array
    content = '{"numbers": [1, 2, 3]}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"numbers": [1, 2, 3]}
    
    # Test case 7: JSON with boolean and null values
    content = '{"flag": true, "value": null}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"flag": True, "value": None}
    
    # Test case 8: JSON with number values
    content = '{"int": 42, "float": 3.14, "exp": 1e5}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "exp": 100000.0}
    
    # Test case 9: JSON with string containing special characters
    content = '{"text": "Hello, \\"world\\"!"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"text": 'Hello, "world"!'}
    
    # Test case 10: JSON with whitespace
    content = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    print("All test cases passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #16
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  # pragma: no cover
    # Test empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.code == "parse_error"

    print("All tests passed!")



# LLM-generated content at query #17
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 2 passed")

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    print("Test case 3 passed")

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    print("Test case 4 passed")

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    print("Test case 5 passed")

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    print("Test case 6 passed")

    # Test case 7: Invalid JSON (missing closing brace)
    content = '{"name": "John"'
    try:
        token = tokenize_json(content)
        print("Test case 7 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "parse_error"
        print("Test case 7 passed")

    # Test case 8: Empty string
    content = ''
    try:
        token = tokenize_json(content)
        print("Test case 8 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "no_content"
        print("Test case 8 passed")

    # Test case 9: Bytes input
    content = b'{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 9 passed")

    # Test case 10: Nested JSON object
    content = '{"person": {"name": "Alice", "age": 25}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "Alice", "age": 25}}
    print("Test case 10 passed")

    # Test case 11: JSON with whitespace
    content = '  {  "key"  :  "value"  }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 11 passed")

    # Test case 12: JSON with escaped characters
    content = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}
    print("Test case 12 passed")

    # Test case 13: JSON with floating point number
    content = '3.14'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    print("Test case 13 passed")

    # Test case 14: JSON with scientific notation
    content = '1.23e-4'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    print("Test case 14 passed")

    # Test case 15: JSON with array of mixed types
    content = '[1, "two", true, null]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", True, None]
    print("Test case 15 passed")

    print("All test cases passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #18
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with valid JSON
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, content)}
    
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test with invalid JSON
    try:
        tokenize_json("{invalid}")
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test with bytes
    content = b'{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    
    print("All tests passed!")



# LLM-generated content at query #19
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Empty string  
    try:  
        tokenize_json("")  
    except ParseError as e:  
        assert e.text == "No content."  
        assert e.code == "no_content"  
        assert e.position.column_no == 1  
        assert e.position.line_no == 1  
        assert e.position.char_index == 0  
    else:  
        assert False, "Expected ParseError for empty string"  
  
    # Test case 2: Valid JSON object  
    token = tokenize_json('{"key": "value"}')  
    assert isinstance(token, DictToken)  
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}  
  
    # Test case 3: Valid JSON array  
    token = tokenize_json('[1, 2, 3]')  
    assert isinstance(token, ListToken)  
    assert token.value == [  
        ScalarToken(1, 1, 1, '[1, 2, 3]'),  
        ScalarToken(2, 4, 4, '[1, 2, 3]'),  
        ScalarToken(3, 7, 7, '[1, 2, 3]')  
    ]  
  
    # Test case 4: Invalid JSON  
    try:  
        tokenize_json('{"key": "value"')  
    except ParseError as e:  
        assert e.code == "parse_error"  
    else:  
        assert False, "Expected ParseError for invalid JSON"  
  
    # Test case 5: Bytes input  
    token = tokenize_json(b'{"key": "value"}')  
    assert isinstance(token, DictToken)  
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}  
  
    print("All tests passed!")  
  
# Run the unit test  
test_tokenize_json()


# LLM-generated content at query #20
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 2 passed")

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    print("Test case 3 passed")

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    print("Test case 4 passed")

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    print("Test case 5 passed")

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    print("Test case 6 passed")

    # Test case 7: Empty string (should raise ParseError)
    content = ''
    try:
        token = tokenize_json(content)
        print("Test case 7 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "no_content"
        print("Test case 7 passed")

    # Test case 8: Invalid JSON (should raise ParseError)
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
        print("Test case 8 failed: Expected ParseError")
    except ParseError as e:
        assert e.code == "parse_error"
        print("Test case 8 passed")

    # Test case 9: Bytes input
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 9 passed")

    # Test case 10: Nested JSON object
    content = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "John", "age": 30}, "city": "New York"}
    print("Test case 10 passed")

    # Test case 11: JSON with whitespace
    content = '  { "name" : "John" , "age" : 30 }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 11 passed")

    # Test case 12: JSON with escaped characters
    content = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}
    print("Test case 12 passed")

    # Test case 13: JSON with unicode characters
    content = '{"name": "Jöhn"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "Jöhn"}
    print("Test case 13 passed")

    # Test case 14: JSON with float number
    content = '{"pi": 3.14159}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"pi": 3.14159}
    print("Test case 14 passed")

    # Test case 15: JSON with scientific notation
    content = '{"number": 1.23e4}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"number": 12300.0}
    print("Test case 15 passed")

    # Test case 16: JSON with negative number
    content = '{"temperature": -10}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"temperature": -10}
    print("Test case 16 passed")

    # Test case 17: JSON with empty object
    content = '{}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    print("Test case 17 passed")

    # Test case 18: JSON with empty array
    content = '[]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    print("Test case 18 passed")

    # Test case 19: JSON with mixed types in array
    content = '[1, "two", true, null]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", True, None]
    print("Test case 19 passed")

    # Test case 20: JSON with nested arrays
    content = '[[1, 2], [3, 4]]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    print("Test case 20 passed")

    print("All test cases passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #21
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 3: Empty string
    content = ""
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 5: JSON with nested objects
    content = '{"person": {"name": "John", "age": 30}}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"person": {"name": "John", "age": 30}}
    
    # Test case 6: JSON with array
    content = '{"names": ["John", "Jane"]}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"names": ["John", "Jane"]}
    
    # Test case 7: JSON with boolean and null values
    content = '{"active": true, "value": null}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"active": True, "value": None}
    
    # Test case 8: JSON with number values
    content = '{"integer": 42, "float": 3.14}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"integer": 42, "float": 3.14}
    
    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #22
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 24
        assert e.position.line_no == 1
        assert e.position.char_index == 23
    
    print("All tests passed!")



# LLM-generated content at query #23
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test with invalid JSON
    try:
        tokenize_json("{")
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with number
    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test with boolean
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test with null
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test with array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 3
    assert all(isinstance(t, ScalarToken) for t in token.value)

    # Test with nested structures
    token = tokenize_json('{"nested": {"key": "value"}}')
    assert isinstance(token, DictToken)
    nested = token.value['nested']
    assert isinstance(nested, DictToken)
    assert nested.value == {"key": ScalarToken("value", 20, 26, '{"nested": {"key": "value"}}')}

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #24
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON object
    json_str = '{"name": "John", "age": 30}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON array
    json_str = '[1, 2, 3]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 2 passed")

    # Test case 3: Valid JSON string
    json_str = '"Hello, World!"'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    print("Test case 3 passed")

    # Test case 4: Valid JSON number
    json_str = '42'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    print("Test case 4 passed")

    # Test case 5: Valid JSON boolean
    json_str = 'true'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    print("Test case 5 passed")

    # Test case 6: Valid JSON null
    json_str = 'null'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    print("Test case 6 passed")

    # Test case 7: Empty JSON object
    json_str = '{}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {}
    print("Test case 7 passed")

    # Test case 8: Empty JSON array
    json_str = '[]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == []
    print("Test case 8 passed")

    # Test case 9: Nested JSON object
    json_str = '{"person": {"name": "Alice", "age": 25}}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "Alice", "age": 25}}
    print("Test case 9 passed")

    # Test case 10: Nested JSON array
    json_str = '[[1, 2], [3, 4]]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    print("Test case 10 passed")

    # Test case 11: JSON with whitespace
    json_str = '  {  "key"  :  "value"  }  '
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    print("Test case 11 passed")

    # Test case 12: JSON with escaped characters
    json_str = '"Line 1\\nLine 2"'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "Line 1\nLine 2"
    print("Test case 12 passed")

    # Test case 13: JSON with unicode characters
    json_str = '"Hello, 世界"'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, 世界"
    print("Test case 13 passed")

    # Test case 14: JSON with scientific notation
    json_str = '1.23e-4'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    print("Test case 14 passed")

    # Test case 15: JSON with negative number
    json_str = '-42'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == -42
    print("Test case 15 passed")

    # Test case 16: JSON with floating point number
    json_str = '3.14159'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14159
    print("Test case 16 passed")

    # Test case 17: JSON with empty string
    json_str = '""'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == ""
    print("Test case 17 passed")

    # Test case 18: JSON with special characters in key
    json_str = '{"key-with-dash": "value"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key-with-dash": "value"}
    print("Test case 18 passed")

    # Test case 19: JSON with multiple key-value pairs
    json_str = '{"a": 1, "b": 2, "c": 3}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"a": 1, "b": 2, "c": 3}
    print("Test case 19 passed")

    # Test case 20: JSON with mixed types in array
    json_str = '[1, "two", true, null]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", True, None]
    print("Test case 20 passed")

    # Test case 21: JSON with deeply nested structure
    json_str = '{"a": {"b": {"c": {"d": "value"}}}}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"a": {"b": {"c": {"d": "value"}}}}
    print("Test case 21 passed")

    # Test case 22: JSON with array of objects
    json_str = '[{"id": 1}, {"id": 2}, {"id": 3}]'
    token = tokenize_json(json_str)
    assert isinstance(token, ListToken)
    assert token.value == [{"id": 1}, {"id": 2}, {"id": 3}]
    print("Test case 22 passed")

    # Test case 23: JSON with object containing array
    json_str = '{"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}
    print("Test case 23 passed")

    # Test case 24: JSON with large integer
    json_str = '12345678901234567890'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 12345678901234567890
    print("Test case 24 passed")

    # Test case 25: JSON with zero
    json_str = '0'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 0
    print("Test case 25 passed")

    # Test case 26: JSON with negative zero
    json_str = '-0'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 0
    print("Test case 26 passed")

    # Test case 27: JSON with exponent only
    json_str = '1e3'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 1000.0
    print("Test case 27 passed")

    # Test case 28: JSON with negative exponent
    json_str = '1e-3'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 0.001
    print("Test case 28 passed")

    # Test case 29: JSON with positive exponent
    json_str = '1e+3'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 1000.0
    print("Test case 29 passed")

    # Test case 30: JSON with decimal and exponent
    json_str = '1.23e2'
    token = tokenize_json(json_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 123.0
    print("Test case 30 passed")

    # Test case 31: JSON with trailing decimal point (should fail)
    json_str = '1.'
    try:
        token = tokenize_json(json_str)
        print("Test case 31 failed: Should have raised ParseError")
    except ParseError:
        print("Test case 31 passed")

    # Test case 32: JSON with leading decimal point (should fail)
    json_str = '.1'
    try:
        token = tokenize_json(json_str)
        print("Test


# LLM-generated content at query #25
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    print("All tests passed!")



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON object
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 2: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 3: Valid JSON string
    content = '"Hello, World!"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 4: Valid JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 5: Valid JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 6: Valid JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 7: Empty JSON object
    content = '{}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 8: Empty JSON array
    content = '[]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 9: Nested JSON object
    content = '{"person": {"name": "Alice", "age": 25}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "Alice", "age": 25}}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 10: Nested JSON array
    content = '[[1, 2], [3, 4]]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [[1, 2], [3, 4]]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 11: JSON with whitespace
    content = '  { "key" : "value" }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 2
    assert token.end == len(content) - 3

    # Test case 12: JSON with escaped characters
    content = '"Hello\\nWorld"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello\nWorld"
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 13: JSON with unicode characters
    content = '"Hello \u0041"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello A"
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 14: JSON with scientific notation
    content = '1.23e-4'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 1.23e-4
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 15: JSON with negative number
    content = '-42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == -42
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 16: JSON with floating point number
    content = '3.14'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 17: JSON with empty string
    content = '""'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == ""
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 18: JSON with special characters in key
    content = '{"key-with-dash": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key-with-dash": "value"}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 19: JSON with nested arrays and objects
    content = '{"array": [1, 2, {"nested": "object"}]}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"array": [1, 2, {"nested": "object"}]}
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 20: JSON with large integer
    content = '12345678901234567890'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 12345678901234567890
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 21: JSON with large negative integer
    content = '-12345678901234567890'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == -12345678901234567890
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 22: JSON with large floating point number
    content = '1.2345678901234567e+308'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 1.2345678901234567e+308
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 23: JSON with large negative floating point number
    content = '-1.2345678901234567e+308'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == -1.2345678901234567e+308
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 24: JSON with escaped quotes in string
    content = '"He said, \\"Hello\\""'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 'He said, "Hello"'
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 25: JSON with backslashes in string
    content = '"C:\\\\path\\\\to\\\\file"'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == "C:\\path\\to\\file"
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 26: JSON with mixed types in array
    content = '[1, "two", true, null, {"key": "value"}]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, "two", True, None, {"key": "value"}]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 27: JSON with empty object in array
    content = '[{}]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [{}]
    assert token.start == 0
    assert token.end == len(content) - 1

    # Test case 28: JSON with empty array in object
    content = '{"empty": []}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"empty": []}
    assert token.start == 0
    assert token.end ==


# LLM-generated content at query #2
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, content)}
    
    # Test with empty string
    content = ''
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON
    content = '{"key": "value"'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15
    
    # Test with bytes
    content = b'{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, content.decode())}
    
    # Test with nested JSON
    content = '{"key": {"nested": "value"}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], DictToken)
    assert token.value["key"].value == {"nested": ScalarToken("value", 17, 23, content)}
    
    # Test with array JSON
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, content), ScalarToken(2, 4, 4, content), ScalarToken(3, 7, 7, content)]
    
    # Test with boolean JSON
    content = '{"true": true, "false": false}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"true": ScalarToken(True, 9, 12, content), "false": ScalarToken(False, 23, 27, content)}
    
    # Test with null JSON
    content = '{"null": null}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"null": ScalarToken(None, 9, 12, content)}
    
    # Test with number JSON
    content = '{"int": 42, "float": 3.14}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"int": ScalarToken(42, 8, 9, content), "float": ScalarToken(3.14, 20, 23, content)}
    
    # Test with whitespace
    content = '  {  "key"  :  "value"  }  '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 15, 21, content)}
    
    print("All tests passed!")



# LLM-generated content at query #3
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Empty string  
    try:  
        tokenize_json("")  
        assert False, "Expected ParseError for empty string"  
    except ParseError as e:  
        assert e.text == "No content."  
        assert e.code == "no_content"  
        assert e.position.column_no == 1  
        assert e.position.line_no == 1  
        assert e.position.char_index == 0  
  
    # Test case 2: Valid JSON object  
    token = tokenize_json('{"key": "value"}')  
    assert isinstance(token, DictToken)  
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}  
  
    # Test case 3: Valid JSON array  
    token = tokenize_json('[1, 2, 3]')  
    assert isinstance(token, ListToken)  
    assert token.value == [  
        ScalarToken(1, 1, 1, '[1, 2, 3]'),  
        ScalarToken(2, 4, 4, '[1, 2, 3]'),  
        ScalarToken(3, 7, 7, '[1, 2, 3]')  
    ]  
  
    # Test case 4: Invalid JSON  
    try:  
        tokenize_json('{"key": "value"')  
        assert False, "Expected ParseError for invalid JSON"  
    except ParseError as e:  
        assert e.code == "parse_error"  
  
    # Test case 5: Bytes input  
    token = tokenize_json(b'{"key": "value"}')  
    assert isinstance(token, DictToken)  
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}  
  
    # Test case 6: JSON with whitespace  
    token = tokenize_json('  { "key" : "value" }  ')  
    assert isinstance(token, DictToken)  
    assert token.value == {"key": ScalarToken("value", 11, 17, '  { "key" : "value" }  ')}  
  
    # Test case 7: JSON with nested structures  
    token = tokenize_json('{"key": [1, 2, {"nested": "value"}]}')  
    assert isinstance(token, DictToken)  
    nested_dict = token.value["key"].value[2]  
    assert isinstance(nested_dict, DictToken)  
    assert nested_dict.value == {"nested": ScalarToken("value", 28, 34, '{"key": [1, 2, {"nested": "value"}]}')}  
  
    print("All tests passed!")  
  


# LLM-generated content at query #4
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test with valid JSON string
    json_str = '{"name": "John", "age": 30}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    json_bytes = b'{"name": "John", "age": 30}'
    token = tokenize_json(json_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test with invalid JSON string
    invalid_json = '{"name": "John", "age": 30'
    try:
        tokenize_json(invalid_json)
    except ParseError as e:
        assert e.code == "parse_error"
        assert "Expecting" in e.text
    
    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #5
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with valid JSON string
    json_str = '{"name": "John", "age": 30, "city": "New York"}'
    token = tokenize_json(json_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30, "city": "New York"}
    
    # Test with valid JSON bytes
    json_bytes = b'{"name": "John", "age": 30, "city": "New York"}'
    token = tokenize_json(json_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30, "city": "New York"}
    
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON string
    invalid_json = '{"name": "John", "age": 30, "city": "New York"'
    try:
        tokenize_json(invalid_json)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 46
    
    # Test with valid JSON array
    json_array = '[1, 2, 3, 4, 5]'
    token = tokenize_json(json_array)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3, 4, 5]
    
    # Test with valid JSON null
    json_null = 'null'
    token = tokenize_json(json_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test with valid JSON boolean
    json_bool = 'true'
    token = tokenize_json(json_bool)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test with valid JSON number
    json_number = '42'
    token = tokenize_json(json_number)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test with valid JSON float
    json_float = '3.14'
    token = tokenize_json(json_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test with valid JSON string
    json_string = '"Hello, World!"'
    token = tokenize_json(json_string)
    assert isinstance(token, ScalarToken)
    assert token.value == "Hello, World!"
    
    # Test with valid JSON nested object
    json_nested = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    token = tokenize_json(json_nested)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "John", "age": 30}, "city": "New York"}
    
    # Test with valid JSON nested array
    json_nested_array = '{"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}'
    token = tokenize_json(json_nested_array)
    assert isinstance(token, DictToken)
    assert token.value == {"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}
    
    # Test with valid JSON empty object
    json_empty_obj = '{}'
    token = tokenize_json(json_empty_obj)
    assert isinstance(token, DictToken)
    assert token.value == {}
    
    # Test with valid JSON empty array
    json_empty_arr = '[]'
    token = tokenize_json(json_empty_arr)
    assert isinstance(token, ListToken)
    assert token.value == []
    
    # Test with valid JSON whitespace
    json_whitespace = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    token = tokenize_json(json_whitespace)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON escaped characters
    json_escaped = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(json_escaped)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}
    
    # Test with valid JSON unicode
    json_unicode = '{"name": "Jöhn", "city": "München"}'
    token = tokenize_json(json_unicode)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "Jöhn", "city": "München"}
    
    # Test with valid JSON scientific notation
    json_scientific = '{"value": 1.23e4}'
    token = tokenize_json(json_scientific)
    assert isinstance(token, DictToken)
    assert token.value == {"value": 12300.0}
    
    # Test with valid JSON negative number
    json_negative = '{"temperature": -10}'
    token = tokenize_json(json_negative)
    assert isinstance(token, DictToken)
    assert token.value == {"temperature": -10}
    
    # Test with valid JSON mixed types
    json_mixed = '{"int": 42, "float": 3.14, "bool": true, "null": null, "string": "hello", "array": [1, 2, 3], "object": {"key": "value"}}'
    token = tokenize_json(json_mixed)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": true, "null": None, "string": "hello", "array": [1, 2, 3], "object": {"key": "value"}}
    
    # Test with valid JSON large number
    json_large = '{"large": 12345678901234567890}'
    token = tokenize_json(json_large)
    assert isinstance(token, DictToken)
    assert token.value == {"large": 12345678901234567890}
    
    # Test with valid JSON decimal
    json_decimal = '{"pi": 3.141592653589793}'
    token = tokenize_json(json_decimal)
    assert isinstance(token, DictToken)
    assert token.value == {"pi": 3.141592653589793}
    
    # Test with valid JSON escaped quotes
    json_escaped_quotes = '{"quote": "He said, \\"Hello\\""}'
    token = tokenize_json(json_escaped_quotes)
    assert isinstance(token, DictToken)
    assert token.value == {"quote": 'He said, "Hello"'}
    
    # Test with valid JSON backslash
    json_backslash = '{"path": "C:\\\\Windows\\\\System32"}'
    token = tokenize_json(json_backslash)
    assert isinstance(token, DictToken)
    assert token.value == {"path": "C:\\Windows\\System32"}
    
    # Test with valid JSON tab
    json_tab = '{"text": "Hello\\tWorld"}'
    token = tokenize_json(json_tab)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello\tWorld"}
    
    # Test with valid JSON carriage return
    json_cr = '{"text": "Hello\\rWorld"}'
    token = tokenize_json(json_cr)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello\rWorld"}
    
    # Test with valid JSON form feed
    json_ff = '{"text": "Hello\\fWorld"}'
    token = tokenize_json(json_ff)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello\fWorld"}
    
    # Test with valid JSON backspace
    json_bs = '{"text": "Hello\\bWorld"}'
    token = tokenize_json(json_bs)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello\bWorld"}
    
    # Test with valid JSON unicode escape
    json_unicode_escape = '{"star": "\\u2605"}'
    token = tokenize_json(json_unicode_escape)
    assert isinstance(token, DictToken)
    assert token.value == {"star": "★"}
    
    # Test with valid JSON surrogate pair
    json_surrogate = '{"emoji": "\\uD83D\\uDE00"}'
    token = tokenize_json(json_surrogate)
    assert isinstance(token, DictToken)
    assert token.value == {"emoji": "😀"}
    
    # Test with valid JSON empty string value
    json_empty_string = '{"name": ""}'
    token = tokenize_json(json_empty_string)
    assert isinstance(token, DictToken)
    assert token.value == {"name": ""}
    
    # Test with valid JSON zero
    json_zero = '{"zero": 0}'
    token = tokenize_json(json_zero)
    assert isinstance(token, DictToken)
    assert token.value == {"zero


# LLM-generated content at query #6
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError for empty string"

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [
        ScalarToken(1, 1, 1, '[1, 2, 3]'),
        ScalarToken(2, 4, 4, '[1, 2, 3]'),
        ScalarToken(3, 7, 7, '[1, 2, 3]')
    ]

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15
    else:
        assert False, "Expected ParseError for invalid JSON"

    # Test case 5: JSON with nested structures
    token = tokenize_json('{"nested": {"inner": [1, 2]}}')
    assert isinstance(token, DictToken)
    nested = token.value["nested"]
    assert isinstance(nested, DictToken)
    inner = nested.value["inner"]
    assert isinstance(inner, ListToken)
    assert inner.value == [
        ScalarToken(1, 23, 23, '{"nested": {"inner": [1, 2]}}'),
        ScalarToken(2, 26, 26, '{"nested": {"inner": [1, 2]}}')
    ]

    # Test case 6: JSON with boolean and null values
    token = tokenize_json('{"bool": true, "null": null}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "bool": ScalarToken(True, 9, 12, '{"bool": true, "null": null}'),
        "null": ScalarToken(None, 22, 25, '{"bool": true, "null": null}')
    }

    # Test case 7: JSON with numbers
    token = tokenize_json('{"int": 42, "float": 3.14}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "int": ScalarToken(42, 8, 9, '{"int": 42, "float": 3.14}'),
        "float": ScalarToken(3.14, 20, 24, '{"int": 42, "float": 3.14}')
    }

    # Test case 8: Bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #7
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 3: Empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
    
    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "Expecting ',' delimiter or '}':."
    
    # Test case 5: JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    
    # Test case 6: JSON string with special characters
    content = '{"message": "Hello, \\"world\\"!"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"message": 'Hello, "world"!'}
    
    # Test case 7: JSON number
    content = '42'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    
    # Test case 8: JSON boolean
    content = 'true'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    # Test case 9: JSON null
    content = 'null'
    token = tokenize_json(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test case 10: JSON with nested objects
    content = '{"person": {"name": "John", "age": 30}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "John", "age": 30}}
    
    # Test case 11: JSON with nested arrays
    content = '{"numbers": [1, 2, [3, 4]]}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"numbers": [1, 2, [3, 4]]}
    
    # Test case 12: JSON with escaped characters
    content = '{"text": "Line 1\\nLine 2"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Line 1\nLine 2"}
    
    # Test case 13: JSON with unicode characters
    content = '{"text": "Hello, 世界"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello, 世界"}
    
    # Test case 14: JSON with scientific notation
    content = '{"value": 1.23e-4}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"value": 0.000123}
    
    # Test case 15: JSON with trailing whitespace
    content = '{"name": "John"}   '
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John"}
    
    # Test case 16: JSON with leading whitespace
    content = '   {"name": "John"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John"}
    
    # Test case 17: JSON with tabs and newlines
    content = '{\n\t"name": "John",\n\t"age": 30\n}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test case 18: JSON with empty object
    content = '{}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {}
    
    # Test case 19: JSON with empty array
    content = '[]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == []
    
    # Test case 20: JSON with multiple spaces in string
    content = '{"text": "Hello   World"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"text": "Hello   World"}
    
    # Test case 21: JSON with backslash in string
    content = '{"path": "C:\\\\Users\\\\John"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"path": "C:\\Users\\John"}
    
    # Test case 22: JSON with mixed types
    content = '{"int": 42, "float": 3.14, "bool": true, "null": null, "string": "hello"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"int": 42, "float": 3.14, "bool": True, "null": None, "string": "hello"}
    
    # Test case 23: JSON with large integer
    content = '{"large": 12345678901234567890}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"large": 12345678901234567890}
    
    # Test case 24: JSON with negative number
    content = '{"negative": -42}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"negative": -42}
    
    # Test case 25: JSON with zero
    content = '{"zero": 0}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"zero": 0}
    
    # Test case 26: JSON with decimal point only
    content = '{"decimal": 0.5}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"decimal": 0.5}
    
    # Test case 27: JSON with exponent only
    content = '{"exponent": 1e3}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"exponent": 1000.0}
    
    # Test case 28: JSON with negative exponent
    content = '{"negative_exponent": 1e-3}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"negative_exponent": 0.001}
    
    # Test case 29: JSON with positive exponent
    content = '{"positive_exponent": 1e+3}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"positive_exponent": 1000.0}
    
    # Test case 30: JSON with capital E exponent
    content = '{"capital_exponent": 1E3}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"capital_exponent": 1000.0}
    
    # Test case 31: JSON with complex number (should fail)
    content = '{"complex": 1+2j}'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 32: JSON with missing closing brace
    content = '{"name": "John"'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 33: JSON with missing comma
    content = '{"name": "John" "age": 30}'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 34: JSON with extra comma
    content = '{"name": "John", "age": 30,}'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 35: JSON with unquoted key
    content = '{name: "John"}'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
    
    # Test case 36: JSON with single quotes
    content = "{'name': 'John'}"
    try:
       


# LLM-generated content at query #8
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    
    # Test with empty string
    content = ''
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test with invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 24
        assert e.position.line_no == 1
        assert e.position.char_index == 23
    
    print("All tests passed!")



# LLM-generated content at query #9
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test with empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError"

    # Test with invalid JSON
    try:
        tokenize_json("{")
    except ParseError as e:
        assert e.text == "Expecting value."
        assert e.code == "parse_error"
        assert e.position.column_no == 2
        assert e.position.line_no == 1
        assert e.position.char_index == 1
    else:
        assert False, "Expected ParseError"

    # Test with valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test with nested JSON
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    nested_token = token.value["key"]
    assert isinstance(nested_token, DictToken)
    assert nested_token.value == {"nested": ScalarToken("value", 20, 26, '{"key": {"nested": "value"}}')}

    # Test with array JSON
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, '[1, 2, 3]'), ScalarToken(2, 4, 4, '[1, 2, 3]'), ScalarToken(3, 7, 7, '[1, 2, 3]')]

    # Test with null, true, false
    token = tokenize_json('{"null": null, "true": true, "false": false}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "null": ScalarToken(None, 9, 12, '{"null": null, "true": true, "false": false}'),
        "true": ScalarToken(True, 22, 25, '{"null": null, "true": true, "false": false}'),
        "false": ScalarToken(False, 36, 40, '{"null": null, "true": true, "false": false}')
    }

    # Test with number
    token = tokenize_json('{"number": 123.45}')
    assert isinstance(token, DictToken)
    assert token.value == {"number": ScalarToken(123.45, 11, 16, '{"number": 123.45}')}

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #10
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError for empty string"

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}
    assert token.start == 0
    assert token.end == 15

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [
        ScalarToken(1, 1, 1, '[1, 2, 3]'),
        ScalarToken(2, 4, 4, '[1, 2, 3]'),
        ScalarToken(3, 7, 7, '[1, 2, 3]')
    ]
    assert token.start == 0
    assert token.end == 9

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15
    else:
        assert False, "Expected ParseError for invalid JSON"

    # Test case 5: Bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 6: JSON with whitespace
    token = tokenize_json('  { "key" : "value" }  ')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 12, 18, '  { "key" : "value" }  ')}

    # Test case 7: JSON with nested structures
    token = tokenize_json('{"nested": {"inner": [1, 2]}}')
    assert isinstance(token, DictToken)
    nested = token.value["nested"]
    assert isinstance(nested, DictToken)
    inner = nested.value["inner"]
    assert isinstance(inner, ListToken)
    assert inner.value == [
        ScalarToken(1, 23, 23, '{"nested": {"inner": [1, 2]}}'),
        ScalarToken(2, 26, 26, '{"nested": {"inner": [1, 2]}}')
    ]

    print("All tests passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #11
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test case 1: Valid JSON string
    content = '{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {'key': ScalarToken('value', 7, 13, content)}
    
    # Test case 2: Valid JSON with nested structure
    content = '{"key": {"nested": "value"}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value['key'], DictToken)
    assert token.value['key'].value == {'nested': ScalarToken('value', 18, 24, content)}
    
    # Test case 3: Valid JSON array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, content), ScalarToken(2, 4, 4, content), ScalarToken(3, 7, 7, content)]
    
    # Test case 4: Valid JSON with boolean and null values
    content = '{"bool": true, "null": null}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {'bool': ScalarToken(True, 9, 12, content), 'null': ScalarToken(None, 21, 24, content)}
    
    # Test case 5: Valid JSON with number values
    content = '{"int": 42, "float": 3.14}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {'int': ScalarToken(42, 8, 9, content), 'float': ScalarToken(3.14, 19, 22, content)}
    
    # Test case 6: Empty string
    content = ''
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
    
    # Test case 7: Invalid JSON string
    content = '{"key": "value"'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
    
    # Test case 8: Bytes input
    content = b'{"key": "value"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {'key': ScalarToken('value', 7, 13, content.decode())}
    
    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #12
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 1 passed")

    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "John", "age": 30}
    print("Test case 2 passed")

    # Test case 3: Empty string
    content = ""
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        print("Test case 3 passed")

    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        token = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        print("Test case 4 passed")

    # Test case 5: JSON with nested objects
    content = '{"person": {"name": "John", "age": 30}}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"person": {"name": "John", "age": 30}}
    print("Test case 5 passed")

    # Test case 6: JSON with array
    content = '[1, 2, 3]'
    token = tokenize_json(content)
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    print("Test case 6 passed")

    # Test case 7: JSON with boolean and null values
    content = '{"active": true, "value": null}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"active": True, "value": None}
    print("Test case 7 passed")

    # Test case 8: JSON with number values
    content = '{"integer": 42, "float": 3.14}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"integer": 42, "float": 3.14}
    print("Test case 8 passed")

    # Test case 9: JSON with escaped characters
    content = '{"message": "Hello\\nWorld"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"message": "Hello\nWorld"}
    print("Test case 9 passed")

    # Test case 10: JSON with unicode characters
    content = '{"name": "José"}'
    token = tokenize_json(content)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "José"}
    print("Test case 10 passed")

    print("All test cases passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #13
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Valid JSON string
    content = '{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 2: Valid JSON bytes
    content = b'{"name": "John", "age": 30}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "age": 30}
    
    # Test case 3: Empty string
    content = ""
    try:
        result = tokenize_json(content)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    
    # Test case 4: Invalid JSON string
    content = '{"name": "John", "age": 30'
    try:
        result = tokenize_json(content)
    except ParseError as e:
        assert e.code == "parse_error"
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.position.column_no == 25
        assert e.position.line_no == 1
        assert e.position.char_index == 24
    
    # Test case 5: JSON string with nested objects
    content = '{"name": "John", "address": {"street": "123 Main St", "city": "New York"}}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "John", "address": {"street": "123 Main St", "city": "New York"}}
    
    # Test case 6: JSON string with array
    content = '{"fruits": ["apple", "banana", "orange"]}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"fruits": ["apple", "banana", "orange"]}
    
    # Test case 7: JSON string with numbers
    content = '{"integer": 42, "float": 3.14, "negative": -10}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"integer": 42, "float": 3.14, "negative": -10}
    
    # Test case 8: JSON string with boolean and null values
    content = '{"flag": true, "empty": null}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"flag": True, "empty": None}
    
    # Test case 9: JSON string with escaped characters
    content = '{"message": "Hello\\nWorld"}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"message": "Hello\nWorld"}
    
    # Test case 10: JSON string with unicode characters
    content = '{"name": "Jöhn", "city": "München"}'
    result = tokenize_json(content)
    assert isinstance(result, DictToken)
    assert result.value == {"name": "Jöhn", "city": "München"}
    
    print("All test cases passed!")

# Run the unit test
test_tokenize_json()


# LLM-generated content at query #14
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0
    else:
        assert False, "Expected ParseError"

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, '[1, 2, 3]'), ScalarToken(2, 4, 4, '[1, 2, 3]'), ScalarToken(3, 7, 7, '[1, 2, 3]')]

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15
    else:
        assert False, "Expected ParseError"

    # Test case 5: JSON with nested objects
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    nested_token = token.value["key"]
    assert isinstance(nested_token, DictToken)
    assert nested_token.value == {"nested": ScalarToken("value", 20, 26, '{"key": {"nested": "value"}}')}

    # Test case 6: JSON with array of objects
    token = tokenize_json('[{"key": "value1"}, {"key": "value2"}]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert isinstance(token.value[0], DictToken)
    assert isinstance(token.value[1], DictToken)
    assert token.value[0].value == {"key": ScalarToken("value1", 8, 15, '[{"key": "value1"}, {"key": "value2"}]')}
    assert token.value[1].value == {"key": ScalarToken("value2", 27, 34, '[{"key": "value1"}, {"key": "value2"}]')}

    # Test case 7: JSON with numbers
    token = tokenize_json('{"int": 42, "float": 3.14, "negative": -10}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "int": ScalarToken(42, 8, 9, '{"int": 42, "float": 3.14, "negative": -10}'),
        "float": ScalarToken(3.14, 19, 22, '{"int": 42, "float": 3.14, "negative": -10}'),
        "negative": ScalarToken(-10, 36, 38, '{"int": 42, "float": 3.14, "negative": -10}')
    }

    # Test case 8: JSON with boolean and null values
    token = tokenize_json('{"true": true, "false": false, "null": null}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "true": ScalarToken(True, 10, 13, '{"true": true, "false": false, "null": null}'),
        "false": ScalarToken(False, 23, 27, '{"true": true, "false": false, "null": null}'),
        "null": ScalarToken(None, 37, 40, '{"true": true, "false": false, "null": null}')
    }

    # Test case 9: JSON with escaped characters
    token = tokenize_json('{"quote": "\\"", "backslash": "\\\\"}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "quote": ScalarToken('"', 11, 13, '{"quote": "\\"", "backslash": "\\\\"}'),
        "backslash": ScalarToken('\\', 29, 31, '{"quote": "\\"", "backslash": "\\\\"}')
    }

    # Test case 10: JSON with unicode characters
    token = tokenize_json('{"emoji": "😀", "text": "café"}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "emoji": ScalarToken("😀", 11, 14, '{"emoji": "😀", "text": "café"}'),
        "text": ScalarToken("café", 24, 29, '{"emoji": "😀", "text": "café"}')
    }

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #15
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, '[1, 2, 3]'), ScalarToken(2, 4, 4, '[1, 2, 3]'), ScalarToken(3, 7, 7, '[1, 2, 3]')]

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"

    # Test case 5: JSON with nested objects
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    nested_token = token.value["key"]
    assert isinstance(nested_token, DictToken)
    assert nested_token.value == {"nested": ScalarToken("value", 20, 26, '{"key": {"nested": "value"}}')}

    # Test case 6: JSON with array of objects
    token = tokenize_json('[{"key": "value1"}, {"key": "value2"}]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert isinstance(token.value[0], DictToken)
    assert isinstance(token.value[1], DictToken)

    # Test case 7: JSON with numbers
    token = tokenize_json('{"int": 42, "float": 3.14}')
    assert isinstance(token, DictToken)
    assert token.value["int"] == ScalarToken(42, 8, 9, '{"int": 42, "float": 3.14}')
    assert token.value["float"] == ScalarToken(3.14, 20, 23, '{"int": 42, "float": 3.14}')

    # Test case 8: JSON with boolean and null
    token = tokenize_json('{"true": true, "false": false, "null": null}')
    assert isinstance(token, DictToken)
    assert token.value["true"] == ScalarToken(True, 9, 12, '{"true": true, "false": false, "null": null}')
    assert token.value["false"] == ScalarToken(False, 23, 27, '{"true": true, "false": false, "null": null}')
    assert token.value["null"] == ScalarToken(None, 36, 39, '{"true": true, "false": false, "null": null}')

    # Test case 9: JSON with escaped characters
    token = tokenize_json('{"key": "value with \\"quotes\\""}')
    assert isinstance(token, DictToken)
    assert token.value["key"] == ScalarToken('value with "quotes"', 8, 30, '{"key": "value with \\"quotes\\""}')

    # Test case 10: JSON with unicode characters
    token = tokenize_json('{"key": "café"}')
    assert isinstance(token, DictToken)
    assert token.value["key"] == ScalarToken('café', 8, 13, '{"key": "café"}')

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #16
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json(): 
    # Test case 1: Empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test case 2: Valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test case 3: Valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [ScalarToken(1, 1, 1, '[1, 2, 3]'), ScalarToken(2, 4, 4, '[1, 2, 3]'), ScalarToken(3, 7, 7, '[1, 2, 3]')]

    # Test case 4: Invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}'."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15

    # Test case 5: JSON with nested objects
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    nested_token = token.value["key"]
    assert isinstance(nested_token, DictToken)
    assert nested_token.value == {"nested": ScalarToken("value", 20, 26, '{"key": {"nested": "value"}}')}

    # Test case 6: JSON with array of objects
    token = tokenize_json('[{"key": "value1"}, {"key": "value2"}]')
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert isinstance(token.value[0], DictToken)
    assert isinstance(token.value[1], DictToken)
    assert token.value[0].value == {"key": ScalarToken("value1", 8, 15, '[{"key": "value1"}, {"key": "value2"}]')}
    assert token.value[1].value == {"key": ScalarToken("value2", 27, 34, '[{"key": "value1"}, {"key": "value2"}]')}

    # Test case 7: JSON with numbers
    token = tokenize_json('{"int": 42, "float": 3.14, "negative": -10}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "int": ScalarToken(42, 8, 9, '{"int": 42, "float": 3.14, "negative": -10}'),
        "float": ScalarToken(3.14, 19, 22, '{"int": 42, "float": 3.14, "negative": -10}'),
        "negative": ScalarToken(-10, 36, 38, '{"int": 42, "float": 3.14, "negative": -10}')
    }

    # Test case 8: JSON with boolean and null values
    token = tokenize_json('{"true": true, "false": false, "null": null}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "true": ScalarToken(True, 10, 13, '{"true": true, "false": false, "null": null}'),
        "false": ScalarToken(False, 24, 28, '{"true": true, "false": false, "null": null}'),
        "null": ScalarToken(None, 38, 41, '{"true": true, "false": false, "null": null}')
    }

    # Test case 9: JSON with escaped characters
    token = tokenize_json('{"key": "value with \\"quotes\\""}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken('value with "quotes"', 8, 30, '{"key": "value with \\"quotes\\""}')}

    # Test case 10: JSON with unicode characters
    token = tokenize_json('{"key": "café"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("café", 8, 13, '{"key": "café"}')}

    # Test case 11: JSON with whitespace
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 16, 22, '  {  "key"  :  "value"  }  ')}

    # Test case 12: JSON with nested arrays
    token = tokenize_json('{"array": [[1, 2], [3, 4]]}')
    assert isinstance(token, DictToken)
    array_token = token.value["array"]
    assert isinstance(array_token, ListToken)
    assert len(array_token.value) == 2
    assert isinstance(array_token.value[0], ListToken)
    assert isinstance(array_token.value[1], ListToken)
    assert array_token.value[0].value == [ScalarToken(1, 12, 12, '{"array": [[1, 2], [3, 4]]}'), ScalarToken(2, 15, 15, '{"array": [[1, 2], [3, 4]]}')]
    assert array_token.value[1].value == [ScalarToken(3, 20, 20, '{"array": [[1, 2], [3, 4]]}'), ScalarToken(4, 23, 23, '{"array": [[1, 2], [3, 4]]}')]

    # Test case 13: JSON with scientific notation
    token = tokenize_json('{"number": 1.23e4}')
    assert isinstance(token, DictToken)
    assert token.value == {"number": ScalarToken(12300.0, 12, 17, '{"number": 1.23e4}')}

    # Test case 14: JSON with trailing comma (should fail)
    try:
        tokenize_json('{"key": "value",}')
    except ParseError as e:
        assert e.text == "Expecting property name enclosed in double quotes."
        assert e.code == "parse_error"

    # Test case 15: JSON with missing comma
    try:
        tokenize_json('{"key1": "value1" "key2": "value2"}')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter."
        assert e.code == "parse_error"

    # Test case 16: JSON with invalid escape
    try:
        tokenize_json('{"key": "\\x"}')
    except ParseError as e:
        assert e.text == "Invalid \\escape."
        assert e.code == "parse_error"

    # Test case 17: JSON with unterminated string
    try:
        tokenize_json('{"key": "value')
    except ParseError as e:
        assert e.text == "Unterminated string starting at."
        assert e.code == "parse_error"

    # Test case 18: JSON with invalid number
    try:
        tokenize_json('{"number": 123.}')
    except ParseError as e:
        assert e.text == "Invalid number."
        assert e.code == "parse_error"

    # Test case 19: JSON with multiple top-level values
    try:
        tokenize_json('{} {}')
    except ParseError as e:
        assert e.text == "Extra data."
        assert e.code == "parse_error"

    # Test case 20: JSON with bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    print("All tests passed!")

# Run the unit tests
test_tokenize_json()


# LLM-generated content at query #17
#--------------------------

# Unit test for function tokenize_json
def test_tokenize_json():  
    # Test empty string
    try:
        tokenize_json("")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test empty bytes
    try:
        tokenize_json(b"")
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.column_no == 1
        assert e.position.line_no == 1
        assert e.position.char_index == 0

    # Test valid JSON
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 7, 13, '{"key": "value"}')}

    # Test invalid JSON
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15

    # Test JSON with number
    token = tokenize_json('{"key": 123}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(123, 7, 9, '{"key": 123}')}

    # Test JSON with boolean
    token = tokenize_json('{"key": true}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(True, 7, 10, '{"key": true}')}

    # Test JSON with null
    token = tokenize_json('{"key": null}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(None, 7, 10, '{"key": null}')}

    # Test JSON with array
    token = tokenize_json('{"key": [1, 2, 3]}')
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], ListToken)
    assert token.value["key"].value == [
        ScalarToken(1, 9, 9, '{"key": [1, 2, 3]}'),
        ScalarToken(2, 12, 12, '{"key": [1, 2, 3]}'),
        ScalarToken(3, 15, 15, '{"key": [1, 2, 3]}'),
    ]

    # Test JSON with nested object
    token = tokenize_json('{"key": {"nested": "value"}}')
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], DictToken)
    assert token.value["key"].value == {
        "nested": ScalarToken("value", 18, 24, '{"key": {"nested": "value"}}')
    }

    # Test JSON with whitespace
    token = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value", 13, 19, '  {  "key"  :  "value"  }  ')}

    # Test JSON with escaped characters
    token = tokenize_json('{"key": "value\\nwith newline"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value\nwith newline", 7, 27, '{"key": "value\\nwith newline"}')}

    # Test JSON with unicode
    token = tokenize_json('{"key": "value with unicode: \u00e9"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken("value with unicode: é", 7, 34, '{"key": "value with unicode: \u00e9"}')}

    # Test JSON with float
    token = tokenize_json('{"key": 123.456}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(123.456, 7, 13, '{"key": 123.456}')}

    # Test JSON with scientific notation
    token = tokenize_json('{"key": 1.23e4}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(12300.0, 7, 12, '{"key": 1.23e4}')}

    # Test JSON with negative number
    token = tokenize_json('{"key": -123}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(-123, 7, 10, '{"key": -123}')}

    # Test JSON with zero
    token = tokenize_json('{"key": 0}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": ScalarToken(0, 7, 7, '{"key": 0}')}

    # Test JSON with empty object
    token = tokenize_json('{}')
    assert isinstance(token, DictToken)
    assert token.value == {}

    # Test JSON with empty array
    token = tokenize_json('[]')
    assert isinstance(token, ListToken)
    assert token.value == []

    # Test JSON with multiple key-value pairs
    token = tokenize_json('{"key1": "value1", "key2": "value2"}')
    assert isinstance(token, DictToken)
    assert token.value == {
        "key1": ScalarToken("value1", 8, 15, '{"key1": "value1", "key2": "value2"}'),
        "key2": ScalarToken("value2", 26, 33, '{"key1": "value1", "key2": "value2"}'),
    }

    # Test JSON with trailing comma (should fail)
    try:
        tokenize_json('{"key": "value",}')
    except ParseError as e:
        assert e.text == "Expecting property name enclosed in double quotes:."
        assert e.code == "parse_error"
        assert e.position.column_no == 17
        assert e.position.line_no == 1
        assert e.position.char_index == 16

    # Test JSON with missing comma
    try:
        tokenize_json('{"key1": "value1" "key2": "value2"}')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter:."
        assert e.code == "parse_error"
        assert e.position.column_no == 18
        assert e.position.line_no == 1
        assert e.position.char_index == 17

    # Test JSON with missing colon
    try:
        tokenize_json('{"key" "value"}')
    except ParseError as e:
        assert e.text == "Expecting ':' delimiter:."
        assert e.code == "parse_error"
        assert e.position.column_no == 8
        assert e.position.line_no == 1
        assert e.position.char_index == 7

    # Test JSON with unclosed object
    try:
        tokenize_json('{"key": "value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or '}':."
        assert e.code == "parse_error"
        assert e.position.column_no == 16
        assert e.position.line_no == 1
        assert e.position.char_index == 15

    # Test JSON with unclosed array
    try:
        tokenize_json('["value"')
    except ParseError as e:
        assert e.text == "Expecting ',' delimiter or ']':."
        assert e.code == "parse_error"
        assert e.position.column_no == 9
        assert e.position.line_no == 1
        assert e.position.char_index == 8

    # Test JSON with invalid escape
    try:
        tokenize_json('{"key": "\\x"}')
    except ParseError as e:
        assert e.text == "Invalid \\escape:."
        assert e.code == "parse_error"
        assert e.position.column_no == 10
        assert e.position.line_no == 1
        assert e.position.char_index == 9

    # Test JSON with invalid number
    try:
        tokenize_json('{"key": 123.}')
    except ParseError as e:
        assert e.text == "Expecting value:."
        assert e.code == "parse_error"
        assert e.position.column_no == 11
        assert e.position.line_no == 1
        assert e.position.char_index == 10

    # Test JSON with invalid true
    try:
        tokenize_json('{"key": tru}')
    except ParseError as e:
        assert e.text == "Expecting value:."
        assert e.code == "parse_error"
        assert e.position.column_no == 11
        assert e.position.line_no == 1
        assert e.position.char_index == 10

    # Test JSON with invalid false
    try:
        tokenize_json('{"key": fals}')
    except ParseError as e:
        assert e.text


