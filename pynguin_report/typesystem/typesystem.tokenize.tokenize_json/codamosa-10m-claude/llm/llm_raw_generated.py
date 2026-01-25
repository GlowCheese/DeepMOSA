####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test JSON with nested structures
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], list)

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test complex nested JSON
    complex_json = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "users" in result.value

    # Test JSON with scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 encoded bytes
    result = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert result.value["arr"][2]["nested"] is True
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("   \n\t  ")
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test unterminated string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "unterminated')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON structure raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{key: "value"}')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #3
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test bytes input with UTF-8
    result = tokenize_json('{"name": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test nested JSON
    result = tokenize_json('{"outer": {"inner": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"] == [1, 2, 3]
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    complex_json = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value["users"]) == 2
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON string
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON scalar values
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert result.value["array"][2]["nested"] is True
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid json}')
    assert exc_info.value.code == "parse_error"
    
    # Test trailing content
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"} extra')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test Token position tracking
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos == len('{"key": "value"}') - 1


# LLM-generated content at query #5
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}], "val": null}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["arr"], ListToken)
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test position information is captured
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos >= 0


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)

    # Test nested JSON objects
    result = tokenize_json('{"outer": {"inner": "value"}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["outer"], DictToken)
    assert result.value["outer"].value == {"inner": "value"}

    # Test nested JSON arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    assert isinstance(result.value[0], ListToken)

    # Test JSON with special characters in string
    result = tokenize_json('{"key": "value with \\"quotes\\""}')
    assert isinstance(result, DictToken)

    # Test JSON with Unicode
    result = tokenize_json('{"name": "José"}')
    assert isinstance(result, DictToken)
    assert result.value["name"] == "José"

    # Test JSON with scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test JSON with negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test malformed JSON with missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON with invalid syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, ]')
    assert exc_info.value.code == "parse_error"


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test token contains position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos is not None


# LLM-generated content at query #8
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}], "str": "test"}')
    assert isinstance(result, DictToken)
    assert result.value["arr"][2]["nested"] is True
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test bytes with UTF-8 encoding
    result = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #9
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test float number
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'

    # Test nested structures
    result = tokenize_json('{"nested": [1, 2, {"deep": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #10
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON with nested structures
    result = tokenize_json('{"nested": [1, 2, {"deep": "value"}]}')
    assert isinstance(result, DictToken)

    # Test JSON from bytes
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test UTF-8 bytes
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'
    assert exc_info.value.position.line_no >= 1
    assert exc_info.value.position.column_no >= 1

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test JSON with negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex nested structure
    complex_json = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "users" in result.value


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test nested structures
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert result.value["nested"][2]["inner"] == "value"
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test complex nested structure with all types
    complex_json = '{"string": "value", "number": 123, "float": 45.67, "bool": true, "null": null, "array": [1, 2], "object": {"nested": "data"}}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert result.value["string"] == "value"
    assert result.value["number"] == 123
    assert result.value["bool"] is True
    assert result.value["null"] is None


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON boolean
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON with float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e5')
    assert isinstance(result, ScalarToken)
    assert result.value == 100000.0

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test bytes input with UTF-8
    result = tokenize_json('{"key": "value"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test nested structures
    result = tokenize_json('{"outer": {"inner": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value == {"outer": {"inner": [1, 2, 3]}}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == "parse_error"

    # Test JSON with complex nested structure
    complex_json = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "users" in result.value
    assert len(result.value["users"]) == 2

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14

    # Test false boolean
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e5')
    assert isinstance(result, ScalarToken)
    assert result.value == 100000.0

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test unclosed object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": true}], "value": null}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON array raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"outer": {"inner": [1, 2, {"nested": true}]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"][2]["nested"] is True

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test unicode bytes input
    result = tokenize_json('{"unicode": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["unicode"] == "café"


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with nested structures
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test complex nested object
    result = tokenize_json('{"a": {"b": {"c": "d"}}}')
    assert isinstance(result, DictToken)

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test false boolean
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}], "val": null}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'
    assert exc_info.value.position.line_no >= 1

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, ')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with missing quotes
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{key: "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes with special characters
    result = tokenize_json(b'{"key": "value with \xc3\xa9"}')
    assert isinstance(result, DictToken)

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #17
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert result.value == {"arr": [1, 2, {"nested": True}]}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON array raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == 'parse_error'
    
    # Test scientific notation
    result = tokenize_json('1e5')
    assert isinstance(result, ScalarToken)
    assert result.value == 100000.0
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value == {"key": "café"}


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON string
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start_pos == 0
    assert token.end_pos == 15

    # Test valid JSON with bytes
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]

    # Test valid JSON scalar (string)
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test valid JSON scalar (number)
    token = tokenize_json('42')
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    # Test valid JSON scalar (float)
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    # Test valid JSON scalar (boolean true)
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test valid JSON scalar (boolean false)
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test valid JSON scalar (null)
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test complex nested structure
    token = tokenize_json('{"array": [1, 2, {"nested": true}], "value": null}')
    assert isinstance(token, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value",}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test UTF-8 bytes decoding
    token = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(token, DictToken)

    # Test invalid UTF-8 bytes (ignored)
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test JSON with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid json}')
    assert exc_info.value.code == 'parse_error'
    assert isinstance(exc_info.value.position, Position)

    # Test nested structures
    result = tokenize_json('{"nested": [1, {"inner": true}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test JSON with scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test JSON with negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex nested structure
    result = tokenize_json('{"a": [1, 2, {"b": "c"}], "d": null}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": ')
    assert exc_info.value.code == 'parse_error'

    # Test nested structures
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)

    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex nested structure with multiple types
    json_str = '{"string": "value", "number": 123, "array": [true, false, null], "nested": {"key": "val"}}'
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert len(result.value) == 4


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test nested JSON structure
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("   \n\t  ")
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)
    
    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == "parse_error"
    
    # Test JSON with scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test nested JSON structure
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value",}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON unclosed brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON unclosed bracket
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure with multiple types
    result = tokenize_json('{"arr": [1, "two", 3.0, true, false, null], "obj": {"nested": "value"}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test UTF-8 bytes decoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,')
    assert exc_info.value.code == 'parse_error'
    
    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["key"], ListToken)
    
    # Test token position tracking
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos == 16
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #24
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test nested structures
    result = tokenize_json('{"array": [1, 2], "nested": {"key": "val"}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"

    # Test trailing content (should fail)
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('123 456')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"data": [{"id": 1, "name": "test"}, {"id": 2}]}')
    assert isinstance(result, DictToken)
    assert "data" in result.value

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test bytes with UTF-8 encoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert token.value == {"key": "value"}
    assert token.start_pos == 0
    assert token.end_pos == 15

    # Test valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert token.value == [1, 2, 3]
    assert token.start_pos == 0

    # Test valid JSON string
    token = tokenize_json('"hello"')
    assert token.value == "hello"

    # Test valid JSON number
    token = tokenize_json('42')
    assert token.value == 42

    # Test valid JSON boolean
    token = tokenize_json('true')
    assert token.value is True

    # Test valid JSON null
    token = tokenize_json('null')
    assert token.value is None

    # Test valid JSON with nested structures
    token = tokenize_json('{"nested": [1, 2, {"key": "value"}]}')
    assert token.value == {"nested": [1, 2, {"key": "value"}]}

    # Test JSON with float
    token = tokenize_json('3.14')
    assert token.value == 3.14

    # Test JSON with scientific notation
    token = tokenize_json('1e10')
    assert token.value == 1e10

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert token.value == {"key": "value"}

    # Test bytes with UTF-8 encoding
    token = tokenize_json('{"key": "value with unicode: \u00e9"}'.encode('utf-8'))
    assert "value with unicode" in token.value["key"]

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure with various types
    complex_json = '{"string": "value", "number": 42, "float": 3.14, "bool": true, "null": null, "array": [1, 2, 3], "nested": {"key": "val"}}'
    token = tokenize_json(complex_json)
    assert token.value["string"] == "value"
    assert token.value["number"] == 42
    assert token.value["float"] == 3.14
    assert token.value["bool"] is True
    assert token.value["null"] is None
    assert token.value["array"] == [1, 2, 3]
    assert token.value["nested"]["key"] == "val"


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start_pos == 0
    assert result.end_pos == 4

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test UTF-8 bytes input
    result = tokenize_json('{"name": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value == {"name": "café"}

    # Test nested JSON
    result = tokenize_json('{"outer": {"inner": 42}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["outer"], DictToken)
    assert result.value["outer"].value == {"inner": 42}

    # Test JSON with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    assert isinstance(exc_info.value.position, Position)

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'

    # Test unterminated JSON string
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "unterminated')
    assert exc_info.value.code == 'parse_error'

    # Test trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14

    # Test complex nested structure
    result = tokenize_json('{"array": [1, "two", {"three": true}], "null_val": null}')
    assert isinstance(result, DictToken)
    assert "array" in result.value
    assert "null_val" in result.value
    assert result.value["null_val"].value is None


# LLM-generated content at query #27
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with_error = None
    try:
        tokenize_json('')
    except ParseError as e:
        with_error = e
    assert with_error is not None
    assert with_error.code == "no_content"
    assert with_error.position.line_no == 1
    assert with_error.position.column_no == 1

    # Test whitespace only string raises ParseError
    with_error = None
    try:
        tokenize_json('   \n\t  ')
    except ParseError as e:
        with_error = e
    assert with_error is not None
    assert with_error.code == "no_content"

    # Test invalid JSON raises ParseError
    with_error = None
    try:
        tokenize_json('{invalid}')
    except ParseError as e:
        with_error = e
    assert with_error is not None
    assert with_error.code == "parse_error"
    assert with_error.position is not None

    # Test complex nested structure
    result = tokenize_json('{"array": [1, "two", {"three": 3}], "null": null}')
    assert isinstance(result, DictToken)
    assert "array" in result.value
    assert "null" in result.value

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14

    # Test UTF-8 bytes
    result = tokenize_json('{"name": "José"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["name"] == "José"

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #28
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested JSON
    result = tokenize_json('{"nested": {"array": [1, 2, {"deep": true}]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"][2]["deep"] is True
    
    # Test JSON with scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4
    
    # Test JSON with negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test token has position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos >= 0
    
    # Test bytes with UTF-8 encoding
    result = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_index == 0
    assert result.end_index == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    assert 'Expecting value' in exc_info.value.text

    # Test invalid JSON - missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1

    # Test JSON with special characters in string
    result = tokenize_json('{"key": "value with \\"quotes\\""}')
    assert isinstance(result, DictToken)

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []

    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"name": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #30
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test JSON with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"
    assert "Expecting value" in exc_info.value.text

    # Test invalid JSON with missing quotes
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{key: "value"}')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test floating point number
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #31
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"outer": {"inner": [1, 2, {"nested": true}]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"][2]["nested"] is True

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, 2, {"inner": "value"}]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["array"], ListToken)
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test token has position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos > 0


# LLM-generated content at query #33
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_index == 0

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test nested JSON structures
    result = tokenize_json('{"arr": [1, 2], "obj": {"nested": true}}')
    assert isinstance(result, DictToken)

    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test with UTF-8 bytes
    result = tokenize_json('{"name": "José"}'.encode('utf-8'))
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == 'parse_error'

    # Test unclosed string
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "unclosed')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    complex_json = '{"users": [{"id": 1, "name": "Alice", "active": true}, {"id": 2, "name": "Bob", "active": false}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #34
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    import pytest
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"array": [1, "two", null], "nested": {"key": true}}')
    assert isinstance(result, DictToken)
    assert "array" in result.value
    assert "nested" in result.value

    # Test JSON with scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test JSON with negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes with special characters
    result = tokenize_json(b'"hello\\u00e9"')
    assert isinstance(result, ScalarToken)


# LLM-generated content at query #35
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    assert exc_info.value.position.line_no >= 1

    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, "two", true, null]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"] == [1, "two", True, None]

    # Test JSON with scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test invalid JSON with missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"key": "café"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "café"}


# LLM-generated content at query #36
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,')
    assert exc_info.value.code == "parse_error"

    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["key"], ListToken)

    # Test complex JSON
    complex_json = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert len(result.value["users"].value) == 2

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "café"


# LLM-generated content at query #37
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key":')
    assert exc_info.value.code == "parse_error"
    
    # Test nested structures
    result = tokenize_json('{"outer": {"inner": [1, 2, {"deep": "value"}]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"][2]["deep"] == "value"
    
    # Test position tracking
    result = tokenize_json('{"a": 1}')
    assert result.start_pos == 0
    assert result.end_pos >= 7
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10


# LLM-generated content at query #38
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test nested structures
    result = tokenize_json('{"nested": {"inner": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["inner"] == [1, 2, 3]
    
    # Test complex JSON with mixed types
    result = tokenize_json('{"str": "value", "num": 42, "bool": true, "null": null, "arr": [1, 2]}')
    assert isinstance(result, DictToken)
    assert result.value["str"] == "value"
    assert result.value["num"] == 42
    assert result.value["bool"] is True
    assert result.value["null"] is None
    assert result.value["arr"] == [1, 2]
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #39
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with trailing content
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid json}')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test token has correct position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos > 0
    assert result.content == '{"key": "value"}'


# LLM-generated content at query #40
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test UTF-8 bytes input
    result = tokenize_json('{"name": "José"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test nested structures
    result = tokenize_json('{"array": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test missing colon raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"
    
    # Test trailing comma raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure with position tracking
    result = tokenize_json('{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}')
    assert isinstance(result, DictToken)
    assert result.start_pos == 0
    assert result.end_pos > result.start_pos


# LLM-generated content at query #41
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, ]')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}], "bool": true}')
    assert isinstance(result, DictToken)
    assert "nested" in result.value
    assert "bool" in result.value

    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes decoding
    result = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["emoji"] == "😀"


# LLM-generated content at query #42
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON string
    token = tokenize_json('{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert isinstance(token, ListToken)
    assert token.value == [1, 2, 3]
    
    # Test valid JSON scalar values
    token = tokenize_json('"hello"')
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    
    token = tokenize_json('123')
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    
    token = tokenize_json('true')
    assert isinstance(token, ScalarToken)
    assert token.value is True
    
    token = tokenize_json('false')
    assert isinstance(token, ScalarToken)
    assert token.value is False
    
    token = tokenize_json('null')
    assert isinstance(token, ScalarToken)
    assert token.value is None
    
    # Test JSON with floats
    token = tokenize_json('3.14')
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    
    # Test JSON with scientific notation
    token = tokenize_json('1e5')
    assert isinstance(token, ScalarToken)
    assert token.value == 100000.0
    
    # Test bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    
    # Test complex nested structure
    token = tokenize_json('{"a": [1, 2, {"b": "c"}]}')
    assert isinstance(token, DictToken)
    assert token.value["a"][2]["b"] == "c"
    
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test unterminated string
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "unterminated')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid number format
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'
    
    # Test bytes with UTF-8 encoding
    token = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(token, DictToken)
    assert token.value["emoji"] == "😀"


# LLM-generated content at query #43
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 14

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number (integer)
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON number (float)
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start_pos == 0
    assert result.end_pos == 4

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["array"], ListToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    assert exc_info.value.position.line_no == 1

    # Test invalid JSON with missing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value" "key2": "value2"}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with whitespace
    result = tokenize_json('  {  "key"  :  "value"  }  ')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test multiline JSON
    result = tokenize_json('{\n  "key": "value"\n}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}


# LLM-generated content at query #44
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_position == 0
    assert result.end_position == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"outer": {"inner": [1, 2, {"deep": "value"}]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"][2]["deep"] == "value"

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test array with mixed types
    result = tokenize_json('[1, "string", true, null, 3.14]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 5


# LLM-generated content at query #45
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test bytes input with UTF-8
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid JSON syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["key"] == [1, 2, 3]
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test Token has position tracking
    result = tokenize_json('{"key": "value"}')
    assert hasattr(result, 'start_position')
    assert hasattr(result, 'end_position')


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test UTF-8 bytes
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, 2, {"key": "value"}]}}')
    assert isinstance(result, DictToken)
    
    # Test token position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos > 0
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #2
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"

    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": ')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}], "value": null}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 2

    # Test negative numbers
    result = tokenize_json('-123')
    assert isinstance(result, ScalarToken)
    assert result.value == -123

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test escaped strings
    result = tokenize_json('"hello\\"world"')
    assert isinstance(result, ScalarToken)
    assert result.value == 'hello"world'

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #3
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, 2, {"deep": "value"}]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"][2]["deep"] == "value"
    
    # Test token position information is captured
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos > 0


# LLM-generated content at query #4
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["key"] == [1, 2, 3]
    
    # Test complex JSON with mixed types
    result = tokenize_json('{"str": "value", "num": 42, "bool": true, "null": null, "arr": [1, "two"]}')
    assert isinstance(result, DictToken)
    assert result.value["str"] == "value"
    assert result.value["num"] == 42
    assert result.value["bool"] is True
    assert result.value["null"] is None
    assert result.value["arr"] == [1, "two"]
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test UTF-8 bytes decoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #5
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_position == 0
    assert result.end_position == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert result.value["arr"][2]["nested"] is True

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"
    assert "Expecting value" in exc_info.value.text

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"
    assert "Expecting ':'" in exc_info.value.text

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == "parse_error"

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14


# LLM-generated content at query #6
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}], "string": "test"}')
    assert isinstance(result, DictToken)
    assert "array" in result.value
    assert "string" in result.value

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e2')
    assert isinstance(result, ScalarToken)
    assert result.value == 150.0

    # Test UTF-8 bytes decoding
    result = tokenize_json('{"emoji": "😀"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["emoji"] == "😀"


# LLM-generated content at query #7
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test complex nested structure
    result = tokenize_json('{"name": "John", "age": 30, "items": [1, 2, 3]}')
    assert isinstance(result, DictToken)
    assert result.value["name"] == "John"
    assert result.value["age"] == 30

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test incomplete JSON
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key":')
    assert exc_info.value.code == 'parse_error'

    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert result is not None
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert result is not None
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test complex nested JSON
    result = tokenize_json('{"nested": {"array": [1, 2, {"deep": true}]}}')
    assert result is not None
    assert isinstance(result, DictToken)

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert result is not None
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position is not None

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test JSON with scientific notation
    result = tokenize_json('1.23e-4')
    assert result is not None
    assert isinstance(result, ScalarToken)

    # Test negative numbers
    result = tokenize_json('-42')
    assert result is not None
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert result is not None
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert result is not None
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #9
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}], "number": 42}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], list)
    
    # Test token position tracking
    result = tokenize_json('{"key": "value"}')
    assert result.start_index == 0
    assert result.end_index > 0
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert token.value == {"key": "value"}
    assert token.start_pos == 0
    assert token.end_pos == 15

    # Test valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert token.value == [1, 2, 3]
    assert token.start_pos == 0
    assert token.end_pos == 8

    # Test valid JSON string
    token = tokenize_json('"hello"')
    assert token.value == "hello"
    assert token.start_pos == 0
    assert token.end_pos == 6

    # Test valid JSON number
    token = tokenize_json('42')
    assert token.value == 42
    assert token.start_pos == 0
    assert token.end_pos == 1

    # Test valid JSON boolean true
    token = tokenize_json('true')
    assert token.value is True

    # Test valid JSON boolean false
    token = tokenize_json('false')
    assert token.value is False

    # Test valid JSON null
    token = tokenize_json('null')
    assert token.value is None

    # Test valid JSON with whitespace
    token = tokenize_json('  { "key" : "value" }  ')
    assert token.value == {"key": "value"}

    # Test valid JSON float
    token = tokenize_json('3.14')
    assert token.value == 3.14

    # Test valid JSON with scientific notation
    token = tokenize_json('1e5')
    assert token.value == 100000.0

    # Test bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert token.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test incomplete JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test incomplete JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value",}')
    assert exc_info.value.code == 'parse_error'

    # Test nested JSON structures
    token = tokenize_json('{"outer": {"inner": [1, 2, {"deep": "value"}]}}')
    assert token.value == {"outer": {"inner": [1, 2, {"deep": "value"}]}}

    # Test JSON with escaped characters
    token = tokenize_json('{"key": "value\\"with\\"quotes"}')
    assert token.value == {"key": 'value"with"quotes'}

    # Test negative numbers
    token = tokenize_json('-42')
    assert token.value == -42

    # Test negative float
    token = tokenize_json('-3.14')
    assert token.value == -3.14


# LLM-generated content at query #11
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}]}')
    assert isinstance(result, DictToken)
    assert result.value["array"][2]["nested"] is True

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": ')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #12
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test nested JSON structure
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert result.value["nested"][2].value["inner"] == "value"

    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test with whitespace
    result = tokenize_json('  {"key": "value"}  ')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex nested structure
    json_str = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    result = tokenize_json(json_str)
    assert isinstance(result, DictToken)
    assert len(result.value["users"]) == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, 2, {"inner": "value"}]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"][2]["inner"] == "value"
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10


# LLM-generated content at query #14
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test JSON with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test JSON with trailing comma raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test nested JSON structures
    result = tokenize_json('{"outer": {"inner": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["outer"], DictToken)
    assert isinstance(result.value["outer"].value["inner"], ListToken)

    # Test JSON with floats
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test JSON with negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 encoded bytes
    result = tokenize_json('{"name": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["name"].value == "café"


# LLM-generated content at query #15
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test JSON with nested structure
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test truncated JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    complex_json = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "users" in result.value

    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #16
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test nested structures
    result = tokenize_json('{"nested": {"array": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["array"], ListToken)
    
    # Test complex JSON
    result = tokenize_json('{"name": "John", "age": 30, "active": true, "balance": null}')
    assert isinstance(result, DictToken)
    assert result.value["name"].value == "John"
    assert result.value["age"].value == 30
    assert result.value["active"].value is True
    assert result.value["balance"].value is None
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #17
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test nested structures
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10


# LLM-generated content at query #18
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2], "nested": {"key": "val"}}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test invalid UTF-8 in bytes (should be ignored)
    result = tokenize_json(b'{"key": "value\xff"}')
    assert isinstance(result, DictToken)

    # Test JSON with scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #19
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"key": "value"}]}')
    assert isinstance(result, DictToken)
    assert result.value["nested"][0].value == 1

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == 'parse_error'
    assert 'Expecting value' in exc_info.value.text

    # Test invalid JSON with missing quotes
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid: "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == 'parse_error'

    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"].value == "café"


# LLM-generated content at query #20
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("")
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json("   \n\t  ")
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, "two", true, null]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"] == [1, "two", True, None]
    
    # Test Token has position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos > 0
    assert result.content == '{"key": "value"}'
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #21
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test bytes input with UTF-8
    result = tokenize_json('{"name": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test unclosed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'
    
    # Test unclosed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    result = tokenize_json('{"outer": {"inner": [1, 2, {"deep": true}]}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"][2]["deep"] is True
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test token position tracking
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos == len('{"key": "value"}') - 1


# LLM-generated content at query #22
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with missing quote
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #23
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with trailing content
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"} extra')
    assert exc_info.value.code == "parse_error"

    # Test nested objects
    result = tokenize_json('{"outer": {"inner": 123}}')
    assert isinstance(result, DictToken)
    assert result.value["outer"]["inner"] == 123

    # Test nested arrays
    result = tokenize_json('[[1, 2], [3, 4]]')
    assert isinstance(result, ListToken)
    assert result.value == [[1, 2], [3, 4]]

    # Test complex structure
    result = tokenize_json('{"arr": [1, "two", null, true], "num": -42.5e2}')
    assert isinstance(result, DictToken)
    assert result.value["arr"] == [1, "two", None, True]
    assert result.value["num"] == -4250.0

    # Test negative number
    result = tokenize_json('-100')
    assert isinstance(result, ScalarToken)
    assert result.value == -100

    # Test scientific notation
    result = tokenize_json('1.5e-3')
    assert isinstance(result, ScalarToken)
    assert result.value == 0.0015

    # Test unicode content
    result = tokenize_json('{"emoji": "🎉"}')
    assert isinstance(result, DictToken)
    assert result.value["emoji"] == "🎉"


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test nested JSON
    result = tokenize_json('{"outer": {"inner": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value == {"outer": {"inner": [1, 2, 3]}}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    assert isinstance(exc_info.value.position, Position)
    
    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": ')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid array raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, ')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    assert "users" in result.value
    
    # Test scientific notation
    result = tokenize_json('1.23e-4')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.23e-4
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}
    
    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #25
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON boolean
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    
    # Test complex nested structure
    result = tokenize_json('{"arr": [1, 2], "obj": {"nested": true}}')
    assert isinstance(result, DictToken)
    
    # Test float number
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test unterminated string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "unterminated')
    assert exc_info.value.code == 'parse_error'
    
    # Test missing colon raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid trailing content raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"} extra')
    assert exc_info.value.code == 'parse_error'
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14


# LLM-generated content at query #26
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, "two", true, null]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["array"], ListToken)
    
    # Test token position tracking
    result = tokenize_json('{"a": 1}')
    assert result.start_position == 0
    assert result.end_position >= 7
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test bytes with UTF-8
    result = tokenize_json('{"key": "value"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #27
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    assert 'Expecting property name' in exc_info.value.text
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, "two", true, null]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"] == [1, "two", True, None]
    
    # Test token position tracking
    result = tokenize_json('{"key": "value"}')
    assert result.start_position == 0
    assert result.end_position >= 0
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test UTF-8 bytes
    result = tokenize_json('{"key": "value with unicode: 你好"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert "你好" in result.value["key"]


# LLM-generated content at query #28
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test missing closing brace raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test invalid number format raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test nested structures
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 1
    assert "nested" in result.value

    # Test complex JSON with mixed types
    result = tokenize_json('{"str": "text", "num": 123, "bool": true, "null": null, "arr": [1, 2], "obj": {}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 6

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []

    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"key": "café"}')
    assert isinstance(result, DictToken)
    assert "café" in str(result.value)


# LLM-generated content at query #29
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number (integer)
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON number (float)
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test nested structures
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"invalid": }')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == "parse_error"
    
    # Test invalid JSON with bad syntax
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test complex nested structure
    complex_json = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test position information is captured
    result = tokenize_json('{"key": "value"}')
    assert hasattr(result, 'start_position')
    assert hasattr(result, 'end_position')


# LLM-generated content at query #30
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    result = tokenize_json('{"nested": {"array": [1, "two", null]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["array"] == [1, "two", None]
    
    # Test token has position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_position == 0
    assert result.end_position > 0
    
    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #31
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 14

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test with bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test nested structure
    result = tokenize_json('{"outer": {"inner": [1, 2]}}')
    assert isinstance(result, DictToken)
    assert result.value == {"outer": {"inner": [1, 2]}}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == "parse_error"

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}')
    assert isinstance(result, DictToken)
    assert len(result.value["users"]) == 2

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #32
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert result.value["nested"][1]["inner"] == "value"
    
    # Test JSON with scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42


# LLM-generated content at query #33
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert result.value == {"nested": [1, {"inner": "value"}]}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test incomplete JSON
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key":')
    assert exc_info.value.code == "parse_error"
    
    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10
    
    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test token has position information
    result = tokenize_json('{"key": "value"}')
    assert result.start_pos == 0
    assert result.end_pos is not None
    
    # Test bytes with UTF-8 encoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value == {"key": "café"}


# LLM-generated content at query #34
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'
    
    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert result.value["nested"]["key"] == [1, 2, 3]
    
    # Test complex nested array
    result = tokenize_json('[{"a": 1}, {"b": 2}]')
    assert isinstance(result, ListToken)
    assert len(result.value) == 2
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test UTF-8 bytes decoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #35
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    
    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    
    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    
    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    
    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    
    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    
    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    
    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    
    # Test nested structures
    result = tokenize_json('{"nested": {"key": [1, 2, 3]}}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], DictToken)
    assert isinstance(result.value["nested"].value["key"], ListToken)
    
    # Test complex JSON with mixed types
    result = tokenize_json('{"string": "value", "number": 42, "bool": true, "null": null, "array": [1, 2], "object": {}}')
    assert isinstance(result, DictToken)
    assert len(result.value) == 6
    
    # Test scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10
    
    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42
    
    # Test UTF-8 bytes decoding
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #36
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'
    assert 'value' in exc_info.value.text.lower()

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with missing colon
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key" "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test complex nested structure
    result = tokenize_json('{"array": [1, 2, {"nested": true}], "string": "test"}')
    assert isinstance(result, DictToken)
    assert "array" in result.value
    assert "string" in result.value

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test scientific notation
    result = tokenize_json('1.5e-10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e-10

    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"key": "café"}')
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #37
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == "no_content"

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no >= 1

    # Test malformed JSON object
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == "parse_error"

    # Test malformed JSON array
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == "parse_error"

    # Test complex nested structure
    result = tokenize_json('{"nested": [1, {"inner": true}], "value": null}')
    assert isinstance(result, DictToken)
    assert "nested" in result.value
    assert isinstance(result.value["nested"], ListToken)

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test UTF-8 bytes with special characters
    result = tokenize_json('{"key": "café"}'.encode('utf-8'))
    assert isinstance(result, DictToken)
    assert result.value["key"] == "café"


# LLM-generated content at query #38
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 15

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]
    assert result.start_pos == 0
    assert result.end_pos == 8

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"
    assert result.start_pos == 0
    assert result.end_pos == 6

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42
    assert result.start_pos == 0
    assert result.end_pos == 1

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False
    assert result.start_pos == 0
    assert result.end_pos == 4

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None
    assert result.start_pos == 0
    assert result.end_pos == 3

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test nested JSON structure
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test JSON with whitespace
    result = tokenize_json('  { "key" : "value" }  ')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test malformed JSON object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": }')
    assert exc_info.value.code == 'parse_error'

    # Test missing closing brace raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test JSON with negative numbers
    result = tokenize_json('-123')
    assert isinstance(result, ScalarToken)
    assert result.value == -123

    # Test JSON with scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []


# LLM-generated content at query #39
#--------------------------

```python
def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test valid JSON with nested structure
    result = tokenize_json('{"nested": [1, 2, {"inner": "value"}]}')
    assert isinstance(result, DictToken)

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON with scientific notation
    result = tokenize_json('1e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1e10

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with trailing comma
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2, 3,]')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON missing closing brace
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value"')
    assert exc_info.value.code == 'parse_error'

    # Test invalid JSON with unquoted key
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{key: "value"}')
    assert exc_info.value.code == 'parse_error'

    # Test negative number
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test negative float
    result = tokenize_json('-3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == -3.14

    # Test complex nested JSON
    complex_json = '{"a": [1, 2, {"b": "c"}], "d": null}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)

    # Test empty object
    result = tokenize_json('{}')
    assert isinstance(result, DictToken)
    assert result.value == {}

    # Test empty array
    result = tokenize_json('[]')
    assert isinstance(result, ListToken)
    assert result.value == []

    # Test unicode characters
    result = tokenize_json('{"key": "こんにちは"}')
    assert isinstance(result, DictToken)

    # Test bytes with UTF-8 encoding
    result = tokenize_json('{"key": "value"}'.encode('utf-8'))
    assert isinstance(result, DictToken)


# LLM-generated content at query #40
#--------------------------

def test_tokenize_json():
    # Test valid JSON object
    result = tokenize_json('{"key": "value"}')
    assert isinstance(result, DictToken)
    assert result.value == {"key": "value"}
    assert result.start_pos == 0
    assert result.end_pos == 16

    # Test valid JSON array
    result = tokenize_json('[1, 2, 3]')
    assert isinstance(result, ListToken)
    assert result.value == [1, 2, 3]

    # Test valid JSON string
    result = tokenize_json('"hello"')
    assert isinstance(result, ScalarToken)
    assert result.value == "hello"

    # Test valid JSON number
    result = tokenize_json('42')
    assert isinstance(result, ScalarToken)
    assert result.value == 42

    # Test valid JSON float
    result = tokenize_json('3.14')
    assert isinstance(result, ScalarToken)
    assert result.value == 3.14

    # Test valid JSON boolean true
    result = tokenize_json('true')
    assert isinstance(result, ScalarToken)
    assert result.value is True

    # Test valid JSON boolean false
    result = tokenize_json('false')
    assert isinstance(result, ScalarToken)
    assert result.value is False

    # Test valid JSON null
    result = tokenize_json('null')
    assert isinstance(result, ScalarToken)
    assert result.value is None

    # Test bytes input
    result = tokenize_json(b'{"key": "value"}')
    assert isinstance(result, DictToken)

    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1

    # Test whitespace only raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n\t  ')
    assert exc_info.value.code == 'no_content'

    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'

    # Test nested structures
    result = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert isinstance(result, DictToken)
    assert isinstance(result.value["nested"], ListToken)

    # Test scientific notation
    result = tokenize_json('1.5e10')
    assert isinstance(result, ScalarToken)
    assert result.value == 1.5e10

    # Test negative numbers
    result = tokenize_json('-42')
    assert isinstance(result, ScalarToken)
    assert result.value == -42

    # Test complex nested JSON
    complex_json = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    result = tokenize_json(complex_json)
    assert isinstance(result, DictToken)


# LLM-generated content at query #41
#--------------------------

```python
import pytest
from typesystem.base import ParseError, Position


def test_tokenize_json():
    # Test valid JSON object
    token = tokenize_json('{"key": "value"}')
    assert token.value == {"key": "value"}
    assert token.start_pos == 0
    
    # Test valid JSON array
    token = tokenize_json('[1, 2, 3]')
    assert token.value == [1, 2, 3]
    assert token.start_pos == 0
    
    # Test valid JSON string
    token = tokenize_json('"hello"')
    assert token.value == "hello"
    
    # Test valid JSON number
    token = tokenize_json('42')
    assert token.value == 42
    
    # Test valid JSON boolean
    token = tokenize_json('true')
    assert token.value is True
    
    token = tokenize_json('false')
    assert token.value is False
    
    # Test valid JSON null
    token = tokenize_json('null')
    assert token.value is None
    
    # Test valid JSON with nested structures
    token = tokenize_json('{"nested": [1, {"inner": "value"}]}')
    assert token.value == {"nested": [1, {"inner": "value"}]}
    
    # Test valid JSON with float
    token = tokenize_json('3.14')
    assert token.value == 3.14
    
    # Test valid JSON with scientific notation
    token = tokenize_json('1.5e10')
    assert token.value == 1.5e10
    
    # Test bytes input
    token = tokenize_json(b'{"key": "value"}')
    assert token.value == {"key": "value"}
    
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('')
    assert exc_info.value.code == 'no_content'
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    
    # Test whitespace-only string raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('   \n  \t  ')
    assert exc_info.value.code == 'no_content'
    
    # Test invalid JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{invalid}')
    assert exc_info.value.code == 'parse_error'
    
    # Test incomplete JSON raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": ')
    assert exc_info.value.code == 'parse_error'
    
    # Test invalid array syntax raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('[1, 2,]')
    assert exc_info.value.code == 'parse_error'
    
    # Test trailing comma in object raises ParseError
    with pytest.raises(ParseError) as exc_info:
        tokenize_json('{"key": "value",}')
    assert exc_info.value.code == 'parse_error'
    
    # Test complex nested JSON
    complex_json = '{"users": [{"id": 1, "name": "Alice", "active": true}, {"id": 2, "name": "Bob", "active": false}]}'
    token = tokenize_json(complex_json)
    assert len(token.value["users"]) == 2
    assert token.value["users"][0]["name"] == "Alice"
    
    # Test JSON with escaped characters
    token = tokenize_json('{"message": "Hello\\nWorld"}')
    assert token.value["message"] == "Hello\nWorld"
    
    # Test negative numbers
    token = tokenize_json('-42')
    assert token.value == -42
    
    token = tokenize_json('[-1, -3.14, -1e5]')
    assert token.value == [-1, -3.14, -1e5]


