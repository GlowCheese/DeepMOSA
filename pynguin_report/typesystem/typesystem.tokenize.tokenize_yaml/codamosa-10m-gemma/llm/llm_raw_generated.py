####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test basic scalar (string)
    content_str = "hello"
    token = tokenize_yaml(content_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    content_int = "123"
    token = tokenize_yaml(content_int)
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    content_float = "12.34"
    token = tokenize_yaml(content_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test boolean
    content_bool = "true"
    token = tokenize_yaml(content_bool)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    content_null = "null"
    token = tokenize_yaml(content_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test List/Sequence
    content_list = "- item1\n- item2"
    token = tokenize_yaml(content_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "itemument2"] # Note: yaml.load handles standard list
    assert token.value == ["item1", "item2"]

    # Test Dictionary/Mapping
    content_dict = "key: value"
    token = tokenize_yaml(content_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test bytes input
    content_bytes = b"foo: bar"
    token = tokenize_yaml(content_bytes)
    assert token.value == {"foo": "bar"}

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test whitespace only error
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test YAML syntax error (ScannerError)
    # Creating an invalid YAML sequence
    invalid_yaml = ": value" 
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position calculation logic via _get_position
    # (Testing the helper indirectly through a multi-line string)
    multi_line = "line1\nline2\nline3"
    # Index of 'l' in 'line2' is 6
    pos = _get_position(multi_line, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion if needed

    # 1. Test valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    active: true
    """
    # Note: tokenize_yaml converts bool to ScalarToken. 
    # Depending on implementation, we expect the parsed value.
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test invalid YAML syntax (ParseError)
    invalid_syntax = """
    name: John Doe
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert "parse_error" in excinfo.value.code

    # 3. Test empty content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # 4. Test validation failure (ValidationError)
    # 'age' is expected to be integer, providing a non-numeric string
    invalid_data_yaml = """
    name: John Doe
    age: not_an_integer
    active: true
    """
    value, errors = validate_yaml(invalid_data_yaml, UserSchema)
    assert errors
    # Check if error is associated with the correct field/position
    # The error message should come from typesystem validation
    assert any("age" in str(e.message) for e in errors)

    # 5. Test bytes input
    bytes_yaml = b"name: Byte User\nage: 25\nactive: false"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"] == "Byte User"

    # 6. Test complex types (List/Dict)
    class ListSchema(Schema):
        items: String
        metadata: String

    complex_yaml = """
    items:
      - item1
      - item2
    metadata:
      key: value
    """
    # Since tokenize_yaml returns ListToken and DictToken, 
    # validate_with_positions will process them.
    # This test assumes the validator is compatible with the token types produced.
    value, errors = validate_yaml(complex_yaml, UserSchema)
    assert not errors
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to check scalar conversion

    # 1. Test valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert isinstance(value["age"], int)

    # 2. Test valid YAML with bytes input
    valid_yaml_bytes = b"name: 'Jane'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert not errors
    assert value["name"] == "Jane"
    assert value["age"] == 25

    # 3. Test validation error (Type mismatch)
    invalid_types_yaml = """
    name: 123
    age: "not_an_int"
    active: "true"
    """
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert errors
    # Check if error contains information about the failed field
    assert any("name" in str(e) or "age" in str(e) for e in errors)

    # 4. Test YAML syntax error (ParseError)
    syntax_error_yaml = """
    name: "John"
    age: : : invalid
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position.line_no == 3

    # 5. Test Empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # 6. Test List/Sequence validation
    class ListSchema(Schema):
        items: String
        
    list_yaml = """
    items:
      - apple
      - banana
    """
    # Note: tokenize_yaml returns ListToken for top level sequences
    # If the YAML is just a list, we test it directly
    list_content = "- apple\n- banana"
    value, errors = validate_yaml(list_content, Schema) # Using base schema for list
    # In typesystem, validating a list against a schema usually requires a specific structure
    # Here we just ensure the tokenizer produces a ListToken correctly
    value, errors = validate_yaml(list_content, String) 
    # This will fail validation because a list is not a string, but tests the flow
    assert errors

    # 7. Test Null handling
    null_yaml = """
    name: null
    age: 40
    active: "true"
    """
    value, errors = validate_yaml(null_yaml, UserSchema)
    assert not errors
    assert value["name"] is None
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test empty string case
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test whitespace only case
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test scalar token (string)
    content_str = "hello"
    token = tokenize_yaml(content_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test scalar token (int)
    content_int = "123"
    token = tokenize_yaml(content_int)
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test scalar token (float)
    content_float = "12.34"
    token = tokenize_yaml(content_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test scalar token (bool)
    content_bool = "true"
    token = tokenize_yaml(content_bool)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test scalar token (null)
    content_null = "null"
    token = tokenize_yaml(content_null)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test sequence token (list)
    content_list = "- item1\n- item2"
    token = tokenize_yaml(content_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    # end_mark.index is at the very end, so end = end_mark.index - 1
    assert token.end == len(content_list.encode('utf-8')) - 1 

    # Test mapping token (dict)
    content_dict = "key: value"
    token = tokenize_yaml(content_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test bytes input
    content_bytes = b"key: value"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test YAML syntax error
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test _get_position helper
    content_multi_line = "line1\nline2\nline3"
    # index at 'l' in line2 is 6
    pos = _get_position(content_multi_line, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # Test _get_position helper with mid-line index
    pos_mid = _get_position(content_multi_line, 7)
    assert pos_mid.line_no == 2
    assert pos_mid.column_no == 2
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion if needed

    # 1. Test Valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    active: true
    """
    # Note: tokenize_yaml converts bool to ScalarToken with value True. 
    # Since we are validating against a Schema, we expect the value to match.
    # Because the custom loader returns tokens, validate_with_positions 
    # processes the token tree.
    
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value['name'] == "John Doe"
    assert value['age'] == 30
    assert value['active'] is True

    # 2. Test Valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane Doe\nage: 25\nactive: false"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert not errors
    assert value['name'] == "Jane Doe"
    assert value['age'] == 25
    assert value['active'] is False

    # 3. Test Invalid YAML Syntax (ParseError)
    invalid_syntax = """
    name: John
    age: : : : 
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # 4. Test Empty Content (No Content Error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test Schema Validation Failure (Validation Error)
    invalid_schema_yaml = """
    name: John
    age: not_an_integer
    active: true
    """
    # validate_with_positions returns (value, errors)
    # If validation fails, errors will contain the error messages
    value, errors = validate_yaml(invalid_schema_yaml, UserSchema)
    assert errors
    # The error should point to the 'age' field
    assert any("age" in str(e.position) for e in errors)

    # 6. Test List/Sequence Tokenization
    class ListSchema(Schema):
        items = String() # This is a simplification, let's use a simple list check

    list_yaml = "- item1\n- item2"
    # We test if the ListToken is correctly produced and traversed
    # We use a simple approach: validate a list of strings
    from typesystem import List
    list_validator = List(String())
    value, errors = validate_yaml(list_yaml, list_validator)
    assert not errors
    assert value == ["item1", "item2"]

def test_get_position():
    content = "line1\nline2\nline3"
    # index of 'l' in 'line2' is 6
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar tokenization

    # 1. Test Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert len(errors) == 0
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert value["active"] == "true"

    # 2. Test Valid List/Sequence
    list_yaml = "- item1\n- item2"
    # Using a simple field for list validation
    from typesystem import List
    value, errors = validate_yaml(list_yaml, List(String()))
    assert len(errors) == 0
    assert value == ["item1", "item2"]

    # 3. Test Invalid YAML Syntax (ParserError)
    invalid_syntax_yaml = """
    name: "John
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test Empty Content (no_content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test Validation Failure (Schema mismatch)
    invalid_data_yaml = """
    name: "John Doe"
    age: "not_an_integer"
    active: "true"
    """
    value, errors = validate_yaml(invalid_data_yaml, TestSchema)
    assert len(errors) > 0
    # Check if the error is associated with the correct position/field
    # Note: Depending on typesystem version, error message content varies
    assert any("age" in str(err) for err in errors)

    # 6. Test Bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert len(errors) == 0
    assert value["name"] == "Byte Test"

    # 7. Test Nested Dictionary
    nested_yaml = """
    user:
      name: "Nested"
      age: 40
    active: "true"
    """
    class NestedSchema(Schema):
        user = Schema(
            name=String(),
            age=Integer()
        )
        active=String()

    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert len(errors) == 0
    assert value["user"]["name"] == "Nested"
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

def test_validate_yaml(mocker):
    # Test valid scalar (string)
    schema_str = Schema({"name": String()})
    content_str = "name: John Doe"
    value, errors = validate_yaml(content_str, schema_str)
    assert value == {"name": "John Doe"}
    assert not errors

    # Test valid integer
    schema_int = Schema({"age": Integer()})
    content_int = "age: 30"
    value, errors = validate_yaml(content_int, schema_int)
    assert value == {"age": 30}
    assert not errors

    # Test valid list
    schema_list = Schema({"tags": Schema({"items": ListToken})}) # Note: using basic validation logic
    # Since validate_with_positions works on Tokens, we test the tokenization via validate_yaml
    content_list = "- python\n- testing"
    # Using a simpler schema for list validation
    from typesystem import List
    schema_list_simple = Schema({"items": List(String())})
    # Note: tokenize_yaml returns a ListToken for a top-level sequence
    # To test a top-level list, we use a schema that expects a list
    content_list = "- python\n- testing"
    # Since tokenize_yaml returns the root token (ListToken), we validate it directly
    # But validate_yaml uses validate_with_positions which expects a validator for the root
    value, errors = validate_yaml(content_list, List(String()))
    assert value == ["python", "testing"]
    assert not errors

    # Test validation error (type mismatch)
    schema_err = Schema({"age": Integer()})
    content_err = "age: not_an_int"
    # Note: tokenize_yaml converts 'not_an_int' to a ScalarToken with string value
    # But the YAML loader might parse it as a string. 
    # If the schema expects Integer and gets a string that isn't a digit:
    content_err_type = "age: abc"
    value, errors = validate_yaml(content_err_type, schema_err)
    assert errors

    # Test YAML syntax error (ParseError)
    content_syntax_error = "key: : value"  # Invalid YAML
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content_syntax_error, schema_str)
    assert excinfo.value.code == "parse_error"

    # Test empty content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", schema_str)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    content_bytes = b"name: Jane Doe"
    value, errors = validate_yaml(content_bytes, schema_str)
    assert value == {"name": "Jane Doe"}
    assert not errors

    # Test complex nested structure
    schema_complex = Schema({
        "user": Schema({
            "id": Integer(),
            "active": String() # Using string to avoid bool/int parsing logic complexity in test
        }),
        "roles": List(String())
    })
    content_complex = """
    user:
      id: 1
      active: "true"
    roles:
      - admin
      - editor
    """
    value, errors = validate_yaml(content_complex, schema_complex)
    assert value["user"]["id"] == 1
    assert value["roles"] == ["admin", "editor"]
    assert not errors
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion if needed

    # 1. Test valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert value["active"] == "true"

    # 2. Test valid YAML with different types (int/float/bool/null)
    mixed_yaml = """
    age: 25
    score: 95.5
    is_valid: true
    metadata: null
    """
    # Using a generic schema to check types
    class MixedSchema(Schema):
        age = Integer()
        score = Integer() # This will fail if we expect float, but let's check float/int handling
        is_valid = String()
        metadata = String()

    # Note: tokenize_yaml converts types via constructors. 
    # Let's test a simpler version to ensure no crashes on basic types.
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert not errors

    # 3. Test validation error (Type mismatch)
    invalid_type_yaml = """
    name: "John Doe"
    age: "not_an_integer"
    active: "true"
    """
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert errors
    assert any("age" in str(err) for err in errors)

    # 4. Test YAML syntax error (ParseError)
    syntax_error_yaml = """
    name: "John Doe"
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 5. Test empty content (No content error)
    empty_yaml = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # 6. Test bytes input
    bytes_yaml = b"name: 'Byte Content'\nage: 10\nactive: 'yes'"
    class ByteSchema(Schema):
        name = String()
        age = Integer()
        active = String()
    
    value, errors = validate_yaml(bytes_yaml, ByteSchema)
    assert not errors
    assert value["name"] == "Byte Content"

    # 7. Test List/Sequence validation
    list_yaml = "- item1\n- item2"
    class ListSchema(Schema):
        items = String() # This is a simplification; usually, we'd use a ListField
    
    # Since validate_with_positions is used, we test the tokenization of lists
    class SimpleListSchema(Schema):
        items = String()
    
    # Testing a raw list token directly via the validator
    class ListOfStrings(Schema):
        # We'll use a basic schema that expects a list
        pass 
    
    # Testing sequence tokenization
    seq_yaml = "- 1\n- 2"
    value, errors = validate_yaml(seq_yaml, ListOfStrings) 
    # Note: validate_with_positions behavior depends on the specific implementation of validate_with_positions
    # But for the scope of this function, we ensure it doesn't crash and parses the list token.
    assert isinstance(value, list)
    assert value == [1, 2]
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion behavior

    # 1. Test valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test invalid YAML syntax (ParseError)
    invalid_syntax = """
    name: "John Doe"
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position.line_no == 3

    # 3. Test validation error (Schema validation failure)
    invalid_data_yaml = """
    name: "John Doe"
    age: "not_an_int"
    active: "true"
    """
    value, errors = validate_yaml(invalid_data_yaml, UserSchema)
    assert errors
    # The error should point to the 'age' field
    assert any("age" in str(err) for err in errors)

    # 4. Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 20\nactive: 'false'"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"] == "Byte Test"

    # 6. Test List/Sequence validation
    class ListSchema(Schema):
        items = String()

    list_yaml = """
    items:
      - apple
      - banana
    """
    # Note: tokenize_yaml returns a ListToken for top-level sequences
    # We test if the validator handles the tokenized structure
    value, errors = validate_yaml(list_yaml, Schema()) # Using empty schema for generic list
    assert not errors
    assert isinstance(value, list)
    assert value[0] == "apple"
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()

    # Case 1: Valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    active: true
    """
    # Note: tokenize_yaml converts bool to ScalarToken with value True/False
    # depending on how the constructor is implemented. 
    # In the provided code, construct_bool returns ScalarToken(True).
    # validate_with_positions will compare this against the Schema.
    
    # We use a simple string-based schema for the bool check to avoid type mismatch
    # if the tokenized value is a python bool but the schema expects string.
    # However, let's test a standard valid case first.
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value['name'] == "John Doe"
    assert value['age'] == 30

    # Case 2: Valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25\nactive: false"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert not errors
    assert value['name'] == "Jane"

    # Case 3: Invalid YAML Syntax (ParseError)
    invalid_syntax = """
    name: John
    age: : unexpected_colon
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Case 4: Empty Content (No Content Error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # Case 5: Validation Error (Schema Mismatch)
    # age is a string here, but schema expects Integer
    invalid_schema_yaml = """
    name: John
    age: not_a_number
    active: true
    """
    value, errors = validate_yaml(invalid_schema_yaml, UserSchema)
    assert errors
    # errors will contain the validation error for 'age'
    assert any("age" in str(e) for e in errors)

    # Case 6: Missing Field
    missing_field_yaml = """
    name: John
    """
    value, errors = validate_yaml(missing_field_yaml, UserSchema)
    assert errors
    assert any("age" in str(e) for e in errors)

    # Case 7: List/Sequence Tokenization
    class ListSchema(Schema):
        items = String() # This is a simplification; usually, we'd use a ListField

    # Testing a simple list structure
    list_yaml = "- item1\n- item2"
    # Since we are testing the function provided, we check if it parses the tokens
    # The provided code returns a ListToken for sequences.
    # We test if the parser handles the list structure without crashing.
    try:
        token, errors = validate_yaml(list_yaml, Schema) 
        # If the schema is just Schema, it should pass if structure is valid
        assert not errors
    except Exception as e:
        pytest.fail(f"List tokenization failed: {e}")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String() # Using String to avoid bool/int confusion in simple test

    # Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30, "active": "true"}

    # Case 2: Valid YAML with Bytes
    valid_bytes = b"name: 'Jane'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(valid_bytes, UserSchema)
    assert errors == []
    assert value["name"] == "Jane"

    # Case 3: Invalid YAML Syntax (ParseError)
    invalid_syntax = """
    name: "John
    age: : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Case 4: Empty Content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # Case 5: Validation Error (Type mismatch)
    # 'age' should be an integer, providing a string that isn't a number
    invalid_types = """
    name: "John"
    age: "not_a_number"
    active: "true"
    """
    # Note: tokenize_yaml converts numbers to ScalarToken with int value, 
    # but if the YAML parser sees it as a string, the validator will catch it.
    # In our specific implementation, we use custom constructors for int/float/bool.
    # If we pass a string that cannot be cast to int by the schema:
    value, errors = validate_yaml(invalid_types, UserSchema)
    assert len(errors) > 0
    # The error message should indicate the field 'age' failed
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # Case 6: Missing required field
    missing_field = """
    name: "John"
    active: "true"
    """
    value, errors = validate_yaml(missing_field, UserSchema)
    assert len(errors) > 0
    # Check if error points to the missing field 'age'
    assert any("age" in str(e) for e in errors)

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is 'l' in 'line2'
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 7
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String() # Using string to test scalar tokenization

    # 1. Test successful validation
    valid_yaml = "name: John Doe\nage: 30\nactive: true"
    # Note: tokenize_yaml converts bool to ScalarToken via construct_bool
    # Depending on how typesystem handles the token, we expect success.
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test validation error (type mismatch)
    invalid_types_yaml = "name: John Doe\nage: not_an_int\nactive: true"
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert errors
    # Find the error related to 'age'
    age_errors = [e for e in errors if "age" in e.position.to_string() or "age" in str(e)]
    assert len(errors) > 0

    # 3. Test YAML syntax error (ParseError)
    # Indentation error in YAML
    syntax_error_yaml = "name: John Doe\n  age: 30\n    invalid: :"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test empty content (no_content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test bytes input
    bytes_yaml = b"name: Jane Doe\nage: 25\nactive: false"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert not errors
    assert value["name"] == "Jane Doe"

    # 6. Test List/Sequence tokenization
    class ListSchema(Schema):
        items = String() # This is a simplified test for structure
    
    list_yaml = "- item1\n- item2"
    # Testing if ListToken is created correctly
    # We use a simpler approach: check if we can parse a basic list
    class SimpleListSchema(Schema):
        tags = String()
    
    # Since validate_with_positions is used, we test the structure
    # This part assumes the validator handles the ListToken structure
    tags_yaml = "tags: [python, testing]"
    value, errors = validate_yaml(tags_yaml, TestSchema)
    assert not errors
    assert value["name"] == "John Doe" # reusing logic

    # 7. Test Scalar types (Int, Float, Bool, Null)
    types_yaml = """
    int_val: 10
    float_val: 10.5
    bool_val: true
    null_val: null
    """
    class TypesSchema(Schema):
        int_val = Integer()
        float_val = String() # Float to string to avoid precision/type issues in test
        bool_val = String()
        null_val = String()

    # Note: Because of the way CustomSafeLoader is written, 
    # it maps specific tags to specific Token types.
    # We just want to ensure no exception is thrown during construction.
    value, errors = validate_yaml(types_yaml, TypesSchema)
    assert not errors
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test empty string / whitespace
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   ")
    assert excinfo.value.code == "no_content"

    # Test Scalar (String)
    token = tokenize_yaml("hello world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello world"

    # Test Scalar (Integer)
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test Scalar (Float)
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test Scalar (Boolean)
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test Scalar (Null)
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test Sequence (List)
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test Mapping (Dict)
    token = tokenize_yaml("key: value\nnum: 10")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 10}

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test complex nested structure
    yaml_content = """
    root:
      list:
        - 1
        - 2
      map:
        a: true
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value["root"]["list"] == [1, 2]
    assert token.value["root"]["map"]["a"] is True

    # Test Syntax Error (Invalid YAML)
    # Indentation error or broken syntax
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test Position calculation for error
    invalid_yaml_newline = "key: value\n  broken: : error"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml_newline)
    # The error should be on line 2
    assert excinfo.value.position.line_no == 2
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion

    # Test Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert value["active"] == "true"

    # Test Case 2: Valid YAML with bytes input
    valid_bytes = b"name: 'Jane'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(valid_bytes, TestSchema)
    assert errors == []
    assert value["name"] == "Jane"
    assert value["age"] == 25

    # Test Case 3: Valid YAML with list/sequence
    class ListSchema(Schema):
        items: Schema
    
    class ItemSchema(Schema):
        id: Integer()

    list_yaml = """
    items:
      - id: 1
      - id: 2
    """
    # Note: Since validate_with_positions is used, we test the structure
    # We'll use a simpler approach for the list test
    class SimpleList(Schema):
        ids: Schema
    
    # Testing a simple scalar validation
    class ScalarSchema(Schema):
        val: Integer()
    
    value, errors = validate_yaml("val: 10", ScalarSchema)
    assert errors == []
    assert value["val"] == 10

    # Test Case 4: Invalid YAML Syntax (ParseError)
    invalid_syntax = """
    name: "John"
    age: : : : 
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 5: Empty Content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 6: Validation Error (Schema mismatch)
    invalid_data_yaml = """
    name: "John"
    age: "not_an_integer"
    active: "true"
    """
    value, errors = validate_yaml(invalid_data_yaml, TestSchema)
    assert len(errors) > 0
    # Check if the error points to the correct field
    assert any("age" in str(err) for err in errors)

    # Test Case 7: Missing required field
    missing_field_yaml = """
    name: "John"
    """
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert len(errors) > 0
    assert any("age" in str(err) for err in errors)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, List, Dict

def test_validate_yaml():
    # Test Case 1: Valid simple scalar (String)
    schema_str = String()
    content_str = "hello world"
    value, errors = validate_yaml(content_str, schema_str)
    assert value == "hello world"
    assert not errors

    # Test Case 2: Valid integer
    schema_int = Integer()
    content_int = "123"
    value, errors = validate_yaml(content_int, schema_int)
    assert value == 123
    assert not errors

    # Test Case 3: Valid List of integers
    schema_list = List(Integer())
    content_list = "- 1\n- 2\n- 3"
    value, errors = validate_yaml(content_list, schema_list)
    assert value == [1, 2, 3]
    assert not errors

    # Test Case 4: Valid Dictionary/Mapping
    schema_dict = Dict({
        "name": String(),
        "age": Integer()
    })
    content_dict = "name: John\nage: 30"
    value, errors = validate_yaml(content_dict, schema_dict)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 5: Validation Error (Type mismatch)
    schema_bool = String() # Expecting string, but providing something that isn't
    content_wrong_type = "123" # This is an int, but validator expects string? 
    # Note: tokenize_yaml converts int to ScalarToken(123). 
    # If validator is String, it should pass if typesystem allows coercion, 
    # but let's test a strict failure.
    schema_strict_int = Integer()
    content_bad_int = "not_an_int"
    value, errors = validate_yaml(content_bad_int, schema_strict_int)
    assert errors

    # Test Case 6: YAML Syntax Error (Invalid indentation)
    content_syntax_error = "key:\n  value\n    bad_indent: true"
    # We use a schema that expects a simple dict
    schema_simple = Dict({"key": String()})
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content_syntax_error, schema_simple)
    assert excinfo.value.code == "parse_error"

    # Test Case 7: Empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", schema_str)
    assert excinfo.value.code == "no_content"

    # Test Case 8: Bytes input
    content_bytes = b"foo: bar"
    schema_bytes_dict = Dict({"foo": String()})
    value, errors = validate_yaml(content_bytes, schema_bytes_dict)
    assert value == {"foo": "bar"}
    assert not errors

    # Test Case 9: Complex nested structure
    schema_complex = Schema({
        "users": List(Dict({
            "id": Integer(),
            "active": String() # using string to avoid bool ambiguity in test
        }))
    })
    content_complex = """
    users:
      - id: 1
        active: "true"
      - id: 2
        active: "false"
    """
    value, errors = validate_yaml(content_complex, schema_complex)
    assert value["users"][0]["id"] == 1
    assert value["users"][1]["id"] == 2
    assert not errors
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()

    # Case 1: Valid YAML
    valid_yaml = "name: John\nage: 30\nactive: true"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {
        "name": "John",
        "age": 30,
        "active": "True" # ScalarToken/construct_bool converts to bool, but typesystem handles it
    }
    # Note: Depending on how ScalarToken stores value, it might be bool. 
    # Let's check the actual content of the decoded value.
    assert value["name"] == "John"
    assert value["age"] == 30

    # Case 2: Invalid YAML Syntax (ParseError)
    invalid_syntax = "name: John\nage: : : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, TestSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Case 3: Empty Content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", TestSchema)
    assert excinfo.value.code == "no_content"

    # Case 4: Validation Error (Schema mismatch)
    # 'age' should be an integer, providing a string that isn't a number
    invalid_types_yaml = "name: John\nage: not_a_number\nactive: true"
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert len(errors) > 0
    # The error message should point to the specific field
    assert any("age" in str(err.position) or "age" in str(err) for err in errors)

    # Case 5: Bytes input
    bytes_yaml = b"name: Jane\nage: 25\nactive: false"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert errors == []
    assert value["name"] == "Jane"

    # Case 6: List/Sequence validation
    class ListSchema(Schema):
        items: String
    
    # Test with a simple list tokenization
    list_yaml = "- item1\n- item2"
    # Note: validate_with_positions on a ListToken directly
    # We test if the tokenizer produces a ListToken correctly
    token = tokenize_yaml(list_yaml)
    from typesystem.tokens import ListToken
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using String to test scalar conversion

    # Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30, "active": "true"}

    # Case 2: Valid YAML with List/Sequence
    list_yaml = """
    - item1
    - item2
    """
    # Using a simple list validation (assuming typesystem supports list validation via tokens)
    # Since the code uses validate_with_positions, we test the tokenization logic
    value, errors = validate_yaml(list_yaml, [String()])
    assert errors == []
    assert value == ["item1", "item2"]

    # Case 3: Invalid YAML syntax (ParseError)
    invalid_syntax_yaml = """
    name: "John
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Case 4: Empty content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # Case 5: Validation Error (Schema mismatch)
    # 'age' should be an integer, providing a string that can't be coerced/is wrong type
    invalid_type_yaml = """
    name: "John"
    age: "not_a_number"
    active: "true"
    """
    value, errors = validate_yaml(invalid_type_yaml, UserSchema)
    assert len(errors) > 0
    # The error should point to the 'age' field
    assert any("age" in str(e.position) for e in errors) or len(errors) > 0

    # Case 6: Bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert errors == []
    assert value["name"] == "Byte Test"

    # Case 7: Scalar types (Float/Int/Bool/Null)
    scalar_yaml = """
    float_val: 1.5
    int_val: 10
    bool_val: true
    null_val: null
    """
    class ScalarSchema(Schema):
        float_val = String() # We check if it parses correctly via tokenization
        int_val = Integer()
        bool_val = String()
        null_val = String()

    value, errors = validate_yaml(scalar_yaml, ScalarSchema)
    assert errors == []
    assert value["float_val"] == "1.5"
    assert value["int_val"] == 10
    assert value["bool_val"] == "true"
    assert value["null_val"] is None or value["null_val"] == "" # Depends on how ScalarToken handles None
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    class UserSchema(Schema):
        name = String()
        age = Integer()

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test invalid YAML syntax (ParseError)
    invalid_syntax = "name: : John"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test empty content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test validation error (Schema validation failure)
    invalid_data = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_data, UserSchema)
    assert value is None
    assert len(errors) > 0
    # Check if the error points to the correct field/position
    # Depending on typesystem implementation, error messages usually contain the field name
    assert any("age" in str(err.message) for err in errors)

    # Test list validation
    class ListSchema(Schema):
        items: List(Integer())
    
    valid_list = "items:\n  - 1\n  - 2"
    value, errors = validate_yaml(valid_list, ListSchema)
    assert value == {"items": [1, 2]}
    assert not errors

    invalid_list = "items:\n  - 1\n  - \"abc\""
    value, errors = validate_yaml(invalid_list, ListSchema)
    assert value is None
    assert len(errors) > 0
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test scalar string
    scalar_yaml = "hello"
    token = tokenize_yaml(scalar_token_yaml := scalar_yaml)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    int_yaml = "123"
    token = tokenize_yaml(int_yaml)
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    float_yaml = "45.67"
    token = tokenize_yaml(float_yaml)
    assert isinstance(token, ScalarToken)
    assert token.value == 45.67

    # Test boolean
    bool_yaml = "true"
    token = tokenize_yaml(bool_yaml)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    null_yaml = "null"
    token = tokenize_yaml(null_yaml)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list (sequence)
    list_yaml = "- item1\n- item2"
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test dictionary (mapping)
    dict_yaml = "key: value\nfoo: bar"
    token = tokenize_yaml(dict_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "foo": "bar"}

    # Test bytes input
    bytes_yaml = b"name: test"
    token = tokenize_yaml(bytes_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "test"}

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"

    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   ")
    assert excinfo.value.code == "no_content"

    # Test invalid YAML syntax error
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position tracking for scalars
    multiline_yaml = "first\nsecond"
    token = tokenize_yaml(multiline_yaml)
    assert isinstance(token, ListToken) # This is actually a scalar in some yaml configs, but let's test scalar content
    
    scalar_multiline = "line1\nline2"
    token = tokenize_yaml(scalar_multiline)
    assert token.start < token.end
    assert token.content == scalar_multiline
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test empty whitespace content
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test ScalarToken (String)
    token = tokenize_yaml("hello world")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello world"
    assert token.start == 0
    assert token.end == 10

    # Test ScalarToken (Integer)
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert isinstance(token.value, int)

    # Test ScalarToken (Float)
    token = tokenize_yaml("45.67")
    assert isinstance(token, ScalarToken)
    assert token.value == 45.67
    assert isinstance(token.value, float)

    # Test ScalarToken (Boolean)
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test ScalarToken (Null)
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test ListToken (Sequence)
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test DictToken (Mapping)
    token = tokenize_yaml("key: value\nnum: 10")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 10}
    assert "key" in token.value

    # Test Bytes input
    token = tokenize_yaml(b"name: python")
    assert isinstance(token, DictToken)
    assert token.value == {"name": "python"}

    # Test ParseError on invalid YAML syntax
    invalid_yaml = "key: : value"  # Invalid mapping syntax
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test _get_position helper directly
    pos = _get_position("line1\nline2\nline3", 10) # index of 'n' in line3
    assert pos.line_no == 3
    assert pos.column_no == 2
    assert pos.char_index == 10
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test Case 1: Basic Scalar (String)
    yaml_str = "hello"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test Case 2: Integer
    yaml_str = "123"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test Case 3: Float
    yaml_str = "45.67"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ScalarToken)
    assert token.value == 45.6
    assert token.start == 0
    assert token.end == 5

    # Test Case 4: Boolean
    yaml_str = "true"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test Case 5: Null
    yaml_str = "null"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test Case 6: List (Sequence)
    yaml_str = "- item1\n- item2"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test Case 7: Dictionary (Mapping)
    yaml_str = "key: value"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test Case 8: Bytes input
    yaml_str = b"name: test"
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "test"}

    # Test Case 9: Empty/Whitespace content (Should raise ParseError)
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   ")
    assert excinfo.value.code == "no_content"

    # Test Case 10: Invalid YAML syntax (Should raise ParseError with position)
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test Case 11: Complex Nested Structure
    complex_yaml = """
    list:
      - num: 1
        flag: true
      - num: 2
        flag: false
    """
    token = tokenize_yaml(complex_yaml)
    assert isinstance(token, DictToken)
    assert token.value["list"][0]["num"] == 1
    assert token.value["list"][1]["flag"] is False

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 6 is 'l' in 'line2'
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # Index 12 is 'l' in 'line3'
    pos = _get_position(content, 12)
    assert pos.line_no == 3
    assert pos.column_no == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, List, Dict

def test_validate_yaml():
    # Define a schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()
        tags = List(String())
        meta = Dict(Dict(String(), String()))

    # Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    tags:
      - python
      - developer
    meta:
      location: "New York"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value == {
        "name": "John Doe",
        "age": 30,
        "tags": ["python", "developer"],
        "meta": {"location": "New York"}
    }

    # Case 2: Invalid YAML Syntax (ParseError)
    invalid_syntax_yaml = """
    name: "John Doe"
    age: : : : 
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Case 3: Empty Content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # Case 4: Valid YAML but fails Schema Validation (Validation Error)
    invalid_types_yaml = """
    name: 123
    age: "not_an_int"
    tags: "not_a_list"
    meta: []
    """
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert errors
    # Check that errors contain positional information or specific field names
    error_messages = [str(e) for e in errors]
    assert any("name" in msg for msg in error_messages)
    assert any("age" in msg for msg in error_messages)
    assert any("tags" in msg for msg in error_messages)

    # Case 5: Bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 1\ntags: []\nmeta: {}"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"] == "Byte Test"

    # Case 6: Scalar/Simple types validation
    simple_schema = String()
    value, errors = validate_yaml("Just a string", simple_schema)
    assert not errors
    assert value == "Just a string"

    value, errors = validate_yaml("123", Integer())
    assert not errors
    assert value == 123
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()

    # 1. Test valid YAML
    valid_yaml = "name: John\nage: 30\nactive: true"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30, "active": "true"}

    # 2. Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25\nactive: false"
    value, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert errors == []
    assert value == {"name": "Jane", "age": 25, "active": "false"}

    # 3. Test validation error (type mismatch)
    invalid_type_yaml = "name: John\nage: not_an_int\nactive: true"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert len(errors) > 0
    # Check that the error points to the correct position (line 2)
    assert errors[0].position.line_no == 2

    # 4. Test validation error (missing field)
    missing_field_yaml = "name: John\nactive: true"
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert len(errors) > 0
    assert "age" in str(errors[0])

    # 5. Test YAML syntax error (ParseError)
    syntax_error_yaml = "name: John\n  age: : : invalid"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 6. Test empty content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # 7. Test complex structures (List/Dict)
    complex_yaml = "items:\n  - 1\n  - 2"
    class ListSchema(Schema):
        items = Integer() # This will fail validation because items is a list, 
                          # but demonstrates the tokenizer works for sequences
    
    # Checking the tokenizer works for sequences specifically
    class ListOfIntsSchema(Schema):
        items = Schema({"val": Integer()})
    
    complex_valid = "items:\n  - val: 1\n  - val: 2"
    value, errors = validate_yaml(complex_valid, ListOfIntsSchema)
    assert errors == []
    assert value["items"][0]["val"] == 1
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, List, Dict

def test_validate_yaml():
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        tags = List(String())
        active = String()  # Using string to test scalar conversion if needed

    # Test Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    tags:
      - python
      - testing
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert value["tags"] == ["python", "testing"]

    # Test Case 2: Invalid YAML Syntax (ParserError)
    invalid_syntax_yaml = """
    name: "John Doe"
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 3: Empty Content (no_content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 4: Valid YAML but Schema Validation Failure (Type mismatch)
    invalid_types_yaml = """
    name: "John Doe"
    age: "not_an_integer"
    tags: []
    active: "true"
    """
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert errors
    # Check that the error is associated with the 'age' field
    assert any("age" in str(err.position) or "age" in str(err.message) for err in errors)

    # Test Case 5: Bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 20\ntags: []\nactive: 'false'"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"] == "Byte Test"

    # Test Case 6: Nested Structures (DictToken/ListToken)
    nested_yaml = """
    metadata:
      id: 123
      labels: [internal, secret]
    """
    class NestedSchema(Schema):
        metadata = Dict(
            id=Integer(),
            labels=List(String())
        )
    
    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert not errors
    assert value["metadata"]["id"] == 123
    assert value["metadata"]["labels"] == ["internal", "secret"]
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion if needed

    # 1. Test valid YAML string
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test valid YAML bytes
    valid_yaml_bytes = b"name: 'Jane'\nage: 25\nactive: 'false'"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert not errors
    assert value["name"] == "Jane"
    assert value["age"] == 25

    # 3. Test invalid YAML syntax (ParseError)
    invalid_syntax = """
    name: "John"
    age: : : : 
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test empty content (ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test validation failure (Schema validation error)
    # age is provided as a string that cannot be converted to int by the schema
    invalid_data_type = """
    name: "John"
    age: "not_a_number"
    active: "true"
    """
    value, errors = validate_yaml(invalid_data_type, UserSchema)
    assert errors
    # Check that the error contains information about the 'age' field
    assert any("age" in str(err) for err in errors)

    # 6. Test missing required field
    missing_field_yaml = """
    name: "John"
    active: "true"
    """
    value, errors = validate_yaml(missing_field_yaml, UserSchema)
    assert errors
    assert any("age" in str(err) for err in errors)

    # 7. Test List/Sequence tokenization
    class ListSchema(Schema):
        items = String() # This is a simplification; testing structure
    
    list_yaml = """
    - item1
    - item2
    """
    # Note: tokenize_yaml returns a ListToken for top-level sequences
    # We validate against a simple structure
    value, errors = validate_yaml(list_yaml, typesystem.List(String()))
    assert not errors
    assert value == ["item1", "item2"]

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is the 'l' in 'line2'
    # line1 (0-4), \n (5), line2 (6-10)
    # content.count("\n", 0, 7) is 1. Line no = 2.
    # 7 - content.rfind("\n", 0, 7) -> 7 - 5 = 2.
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 2
    assert pos.char_index == 7
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    class UserSchema(Schema):
        name = String()
        age = Integer()

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML with bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test invalid YAML structure (Syntax Error)
    invalid_syntax_yaml = "name: John\nage: : : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test validation error (Type mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    # Note: tokenize_yaml converts 'not_an_int' to a ScalarToken. 
    # The validation error occurs during the validate_with_positions step.
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert errors
    # Check that the error points to the correct field
    error_messages = [str(e) for e in errors]
    assert any("age" in msg for msg in error_messages)

    # Test list validation
    class ListSchema(Schema):
        items = String()

    list_yaml = "items: [a, b, c]"
    # Since tokenize_yaml returns a ListToken for the root if it's a sequence
    # We need a schema that matches the root token type. 
    # If the root is a ListToken, we validate against a List of Strings.
    from typesystem import List
    list_validator = List(String())
    value, errors = validate_yaml(list_yaml, list_validator)
    assert value == ["a", "b", "c"]
    assert not errors

    # Test complex nested structure
    nested_yaml = """
    user:
      name: Alice
      details:
        active: true
    """
    class NestedSchema(Schema):
        user = Schema({
            "name": String(),
            "details": Schema({
                "active": String() # Using string because bool constructor might be overridden
            })
        })
    
    # Note: The provided tokenize_yaml specifically overrides bool/int/float 
    # to return ScalarTokens. So we validate against String or similar.
    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert value["user"]["name"] == "Alice"
    assert not errors
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar conversion if needed

    # 1. Test Valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    active: true
    """
    # Note: tokenize_yaml converts bool to ScalarToken(True). 
    # Depending on typesystem version, we validate against the expected types.
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert value["active"] is True

    # 2. Test Valid YAML with Bytes
    valid_yaml_bytes = b"name: Jane Doe\nage: 25\nactive: false"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert not errors
    assert value["name"] == "Jane Doe"
    assert value["age"] == 25
    assert value["active"] is False

    # 3. Test Validation Error (Type Mismatch)
    invalid_types_yaml = """
    name: John
    age: not_a_number
    active: true
    """
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert errors
    # Check if the error points to the correct field/type
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # 4. Test YAML Syntax Error (ParseError)
    syntax_error_yaml = """
    name: John
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # 5. Test Empty Content (No Content Error)
    empty_yaml = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, UserSchema)
    assert excinfo.value.code == "no_content"

    # 6. Test List/Sequence Validation
    class ListSchema(Schema):
        items: List(Integer())

    list_yaml = """
    items:
      - 1
      - 2
      - 3
    """
    # We need to define List if not imported, but assuming standard typesystem usage
    from typesystem import List
    value, errors = validate_yaml(list_yaml, ListSchema)
    assert not errors
    assert value == [1, 2, 3]

    # 7. Test Complex Nested Structure
    complex_yaml = """
    user:
      name: Alice
      meta:
        id: 123
    """
    class ComplexSchema(Schema):
        user: Schema({
            "name": String(),
            "meta": Schema({
                "id": Integer()
            })
        })

    value, errors = validate_yaml(complex_yaml, ComplexSchema)
    assert not errors
    assert value["user"]["meta"]["id"] == 123
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Setup schemas
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to test scalar parsing

    # 1. Test Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    assert isinstance(value["name"], ScalarToken)
    assert isinstance(value["age"], ScalarToken)

    # 2. Test Valid List/Sequence
    list_yaml = "- apple\n- banana\n- cherry"
    value, errors = validate_yaml(list_yaml, Schema({"items": ListToken})) # Note: Testing list structure
    # Since validate_yaml uses validate_with_positions, we test against a simple list schema
    class ListSchema(Schema):
        items = ListToken
    
    # Simplified test for list structure
    value, errors = validate_yaml("- apple\n- banana", Schema(ListToken))
    assert not errors
    assert len(value) == 2

    # 3. Test Invalid YAML Syntax (ScannerError)
    invalid_syntax = """
    name: "John
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert excinfo.value.position.line_no == 2

    # 4. Test Empty Content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test Validation Error (Type Mismatch)
    # 'age' should be int, providing a string that can't be parsed as int in the schema context
    # Note: tokenize_yaml converts it to a ScalarToken with value 30, 
    # but if we force a type mismatch in the Schema definition:
    class StrictSchema(Schema):
        age = String()
    
    # Test valid parsing but schema validation failure
    # If we use a field that expects something else
    class WrongTypeSchema(Schema):
        age = Integer()
    
    # Manually constructing a scenario where the token value is valid YAML 
    # but violates the typesystem schema constraints if possible.
    # However, tokenize_yaml handles the conversion. 
    # Let's test a schema mismatch.
    invalid_type_yaml = "age: not_a_number"
    # In tokenize_yaml, 'not_a_number' becomes a ScalarToken with value 'not_a_number'
    # The Integer field in typesystem will fail to validate this string.
    value, errors = validate_yaml(invalid_type_yaml, WrongTypeSchema)
    assert errors
    assert "age" in errors[0].path

    # 6. Test Bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 25\nactive: 'true'"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"].value == "Byte Test"

    # 7. Test Dictionary/Mapping structure
    dict_yaml = "key: value"
    value, errors = validate_yaml(dict_yaml, Schema({"key": String()}))
    assert not errors
    assert value["key"].value == "value"
```


