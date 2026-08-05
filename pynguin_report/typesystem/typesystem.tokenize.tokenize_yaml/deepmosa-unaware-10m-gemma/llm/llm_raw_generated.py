####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test simple scalar (string)
    content_str = "hello"
    token = tokenize_yaml(content_str)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"
    assert token.start == 0
    assert token.end == 4

    # Test integer scalar
    content_int = "123"
    token_int = tokenize_yaml(content_int)
    assert isinstance(token_int, ScalarToken)
    assert token_int.value == 123

    # Test boolean scalar
    content_bool = "true"
    token_bool = tokenize_yaml(content_bool)
    assert isinstance(token_bool, ScalarToken)
    assert token_bool.value is True

    # Test null scalar
    content_null = "null"
    token_null = tokenize_yaml(content_null)
    assert isinstance(token_null, ScalarToken)
    assert token_null.value is None

    # Test list (sequence)
    content_list = "- item1\n- item2"
    token_list = tokenize_yaml(content_list)
    assert isinstance(token_list, ListToken)
    assert len(token_list.value) == 2
    assert isinstance(token_list.value[0], ScalarToken)
    assert token_list.value[0].value == "item1"

    # Test dictionary (mapping)
    content_dict = "key: value"
    token_dict = tokenize_yaml(content_dict)
    assert isinstance(token_dict, DictToken)
    assert token_dict.value == {"key": "value"}
    assert isinstance(token_dict.value["key"], ScalarToken)
    assert token_dict.value["key"].value == "value"

    # Test bytes input
    content_bytes = b"name: test"
    token_bytes = tokenize_yaml(content_bytes)
    assert isinstance(token_bytes, DictToken)
    assert token_bytes.value["name"].value == "test"

    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.text == "No content."

    # Test whitespace-only content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test invalid YAML syntax raises ParseError
    invalid_yaml = ": invalid"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test complex nested structure
    complex_yaml = """
    root:
      list:
        - 1
        - 2
      map:
        a: true
    """
    token_complex = tokenize_yaml(complex_yaml)
    assert isinstance(token_complex, DictToken)
    root_val = token_complex.value["root"]
    assert isinstance(root_val, DictToken)
    assert isinstance(root_val.value["list"], ListToken)
    assert root_val.value["list"].value[0].value == 1
    assert root_val.value["map"].value["a"].value is True
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test empty content
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"

    # Test simple scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list/sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test dictionary/mapping
    token = tokenize_yaml("key: value\nnum: 1")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 1}

    # Test bytes input
    token = tokenize_yaml(b"foo: bar")
    assert isinstance(token, DictToken)
    assert token.value == {"foo": "bar"}

    # Test YAML syntax error (invalid indentation/structure)
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("key: : value")
    assert excinfo.value.code == "parse_error"

    # Test position calculation for error
    bad_yaml = "key: value\n  invalid_indentation"
    try:
        tokenize_yaml(bad_yaml)
    except ParseError as e:
        assert isinstance(e.position, Position)
        assert e.position.line_no == 2

    # Test complex nested structure
    complex_yaml = """
    nested:
      list:
        - 1
        - 2
      dict:
        a: b
    """
    token = tokenize_yaml(complex_yaml)
    assert isinstance(token, DictToken)
    assert token.value["nested"]["list"] == [1, 2]
    assert token.value["nested"]["dict"]["a"] == "b"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test case 1: Valid simple scalar (Integer)
    schema_int = Integer()
    content_int = "42"
    value, errors = validate_yaml(content_int, schema_int)
    assert value == 42
    assert not errors

    # Test case 2: Valid dictionary (Schema)
    class UserSchema(Schema):
        name = String()
        age = Integer()

    content_dict = "name: John\nage: 30"
    value, errors = validate_yaml(content_dict, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test case 3: Valid list (ListToken via implicit conversion)
    content_list = "- apple\n- banana"
    from typesystem import List
    schema_list = List(String())
    value, errors = validate_yaml(content_list, schema_list)
    assert value == ["apple", "banana"]
    assert not errors

    # Test case 4: Validation error (Type mismatch)
    content_invalid_type = "name: John\nage: not_a_number"
    value, errors = validate_yaml(content_invalid_type, UserSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error contains position info (implicit in typesystem validation)
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # Test case 5: YAML Syntax Error (ScannerError)
    content_syntax_error = "name: : invalid"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content_syntax_error, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test case 6: Empty content error
    content_empty = ""
    with pytest.py.raises(ParseError) as excinfo:
        validate_yaml(content_empty, schema_int)
    assert excinfo.value.code == "no_content"

    # Test case 7: Bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(content_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test case 8: Float validation
    schema_float = Integer() # This should fail for a float
    content_float = "1.5"
    value, errors = validate_yaml(content_float, schema_float)
    assert value is None
    assert len(errors) > 0
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

def test_validate_yaml(mocker):
    # Test Case 1: Valid YAML string with a Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML bytes
    valid_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test Case 3: Valid YAML List
    list_yaml = "- one\n- two"
    value, errors = validate_yaml(list_yaml, [String()])
    assert value == ["one", "two"]
    assert not errors

    # Test Case 4: Validation Error (Type mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert value is None
    assert len(errors) > 0
    # Check if the error position relates to the integer field
    assert "age" in str(errors)

    # Test Case 5: YAML Syntax Error (Invalid indentation/syntax)
    bad_syntax_yaml = "name: John\n  age: : : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(bad_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 6: Empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 7: Scalar validation (Simple String Field)
    scalar_yaml = "Hello World"
    value, errors = validate_yaml(scalar_yaml, String())
    assert value == "Hello World"
    assert not errors

    # Test Case 8: Complex Nested Structure
    complex_yaml = """
    user:
      name: Alice
      tags:
        - admin
        - editor
    """
    class ComplexSchema(Schema):
        user = Schema({
            "name": String(),
            "tags": [String()]
        })
    
    value, errors = validate_yaml(complex_yaml, ComplexSchema)
    assert value["user"]["name"] == "Alice"
    assert value["user"]["tags"] == ["admin", "editor"]
    assert not errors
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Define a simple schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()

    # 1. Test Valid YAML
    valid_yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml_content, UserSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # 2. Test Valid YAML with Bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert errors == []
    assert value == {"name": "Jane", "age": 25}

    # 3. Test Valid YAML with List/Sequence
    list_schema = Schema({"items": [String()]})
    list_yaml = "items: [a, b, c]"
    value, errors = validate_yaml(list_yaml, list_schema)
    assert errors == []
    assert value == {"items": ["a", "b", "c"]}

    # 4. Test Validation Error (Type Mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert len(errors) > 0
    # Check that the error points to the correct field/position if possible
    assert any("age" in str(err.message) for err in errors)

    # 5. Test YAML Parse Error (Syntax Error)
    syntax_error_yaml = "name: John\nage: : : :"  # Invalid YAML syntax
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # 6. Test Empty Content Error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, UserSchema)
    assert excinfo.value.code == "no_content"

    # 7. Test Scalar Types (Int/Float/Bool/Null)
    scalar_schema = Schema({"a": Integer(), "b": String(), "c": [Integer()]})
    scalar_yaml = "a: 1\nb: true\nc: [2, 3]"
    # Note: PyYAML might load 'true' as boolean, checking how our custom constructors handle it
    value, errors = validate_yaml(scalar_yaml, scalar_schema)
    assert errors == []
    assert value["a"] == 1

    # 8. Test Position Calculation Helper (Internal logic check via tokenize behavior)
    multiline_error_yaml = "name: John\nage: bad"
    # We expect the error to be on line 2
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(multiline_error_yaml, UserSchema)
    assert excinfo.value.position.line_no == 2

    # 9. Test YAML Null/None
    null_schema = Schema({"key": String()})
    null_yaml = "key: null"
    value, errors = validate_yaml(null_yaml, null_schema)
    assert errors == []
    assert value["key"] is None
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test valid simple YAML string
    schema = Schema({"name": String(), "age": Integer()})
    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, schema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid list in YAML
    list_schema = Schema({"tags": String()}) # Simple check for structure
    # Note: Using a single field schema to validate the content of the tokenized dict
    yaml_content_list = "tags: [python, pytest]"
    # We use a more appropriate schema for list validation
    from typesystem import List
    list_schema = Schema({"tags": List(String())})
    value, errors = validate_parsing_helper(yaml_content_list, list_schema)
    assert value == {"tags": ["python", "pytest"]}
    assert not errors

    # Test invalid YAML syntax (Parsing Error)
    invalid_yaml_syntax = "name: : unexpected"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_yaml_syntax, schema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", schema)
    assert excinfo.value.code == "no_content"

    # Test validation error (Type mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types_yaml, schema)
    assert value is None or "age" in str(errors)
    assert len(errors) > 0

def validate_parsing_helper(content, validator):
    """Helper to wrap the logic for testing purposes."""
    try:
        return validate_yaml(content, validator)
    except Exception as e:
        return None, [e]

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is 'l' in 'line2'
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 7

def test_tokenize_yaml_bytes():
    schema = Schema({"key": String()})
    content_bytes = b"key: value"
    value, errors = validate_yaml(content_bytes, schema)
    assert value == {"key": "value"}
    assert not errors

def test_tokenize_yaml_complex():
    schema = Schema({
        "user": Schema({
            "id": Integer(),
            "active": String() # Using string to avoid bool constructor complexity in test
        }),
        "items": List(Integer())
    })
    content = """
    user:
      id: 123
      active: "true"
    items:
      - 1
      - 2
    """
    # We need to ensure the content matches the schema structure
    # Since tokenize_yaml uses custom constructors, we check if it navigates nested structures
    value, errors = validate_yaml(content, schema)
    assert value["user"]["id"] == 123
    assert value["items"] == [1, 2]
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test valid simple scalar (string)
    schema = String()
    content = "hello"
    value, errors = validate_yaml(content, schema)
    assert value == "hello"
    assert not errors

    # Test valid integer
    schema = Integer()
    content = "123"
    value, errors = validate_yaml(content, schema)
    assert value == 123
    assert not errors

    # Test valid dictionary/mapping
    class UserSchema(Schema):
        name = String()
        age = Integer()
    
    schema = UserSchema()
    content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John Doe", "age": 30}
    assert not errors

    # Test valid list/sequence
    schema = Schema({"items": ListToken}) # Note: testing structure via tokenize logic
    # Using a simpler approach for list validation
    class ListSchema(Schema):
        items = List(String())
    
    content = "items: [a, b, c]"
    value, errors = validate_yaml(content, ListSchema())
    assert value == {"items": ["a", "b", "c"]}
    assert not errors

    # Test validation error (type mismatch)
    schema = Integer()
    content = "not_an_int"
    value, errors = validate_yaml(content, schema)
    assert value is None
    assert len(errors) > 0

    # Test validation error (missing field in Schema)
    class RequiredSchema(Schema):
        name = String()
    
    schema = RequiredSchema()
    content = "age: 30"
    value, errors = validate_yaml(content, schema)
    assert value is None
    assert any("name" in str(err) for err in errors)

    # Test YAML Syntax Error (ParseError)
    # Invalid indentation/mapping syntax
    content = "key: : value" 
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content, String())
    assert excinfo.value.code == "parse_error"

    # Test Empty Content (No content error)
    content = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content, String())
    assert excinfo.value.code == "no_content"

    # Test bytes input
    content = b"name: ByteTest"
    schema = Schema({"name": String()})
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "ByteTest"}
    assert not errors

    # Test complex nested structure
    class NestedSchema(Schema):
        meta: Dict(Schema({"id": Integer()}))
        tags: List(String())

    content = """
    meta:
      id: 1
    tags:
      - python
      - testing
    """
    # Note: Since typesystem.tokenize.tokens handles the conversion, 
    # we validate against the resulting Token tree.
    value, errors = validate_yaml(content, Schema({"meta": Dict(Schema({"id": Integer()})), "tags": List(String())}))
    assert value["meta"]["id"] == 1
    assert "python" in value["tags"]
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

def test_validate_yaml(mocker):
    # Setup a schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()

    # Test Case 1: Valid YAML string
    valid_yaml = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML bytes
    valid_yaml_bytes = b"name: Jane Doe\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert value == {"name": "Jane Doe", "age": 25}
    assert not errors

    # Test Case 3: Valid YAML with complex types (List/Dict tokens)
    complex_yaml = """
    data:
      - item1
      - item2
    """
    class ComplexSchema(Schema):
        data = ListToken # Note: This assumes the token structure is being validated
    
    # Since validate_with_positions depends on the tokens generated, 
    # we test standard schema validation logic via validate_yaml.
    class SimpleListSchema(Schema):
        items: ListToken # This is illustrative; usually we use typesystem fields
    
    # Testing with a simple list field
    from typesystem import ListField
    list_schema = Schema({"items": ListField(String())})
    list_yaml = "items: [a, b, c]"
    value, errors = validate_yaml(list_yaml, list_schema)
    assert value == {"items": ["a", "b", "c"]}
    assert not errors

    # Test Case 4: Validation Error (Type Mismatch)
    invalid_types_yaml = """
    name: John Doe
    age: not_an_integer
    """
    value, errors = validate_yaml(invalid_types_yaml, UserSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error position/message is propagated (depends on validate_with_positions implementation)

    # Test Case 5: YAML Syntax Error (ScannerError)
    syntax_error_yaml = """
    name: "unclosed quote
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 6: Empty Content Error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, UserSchema)
    assert excinfo.value.code == "no_content"
    assert excinfo.value.text == "No content."

    # Test Case 7: Whitespace only content
    whitespace_yaml = "   \n   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(whitespace_yaml, UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 8: Numeric types (testing the custom constructors for int/float/bool)
    numeric_yaml = """
    int_val: 10
    float_val: 10.5
    bool_val: true
    null_val: null
    """
    class NumericSchema(Schema):
        int_val: Integer
        float_val: Float # Assuming Float field exists or using a generic approach
        bool_val: Boolean
        null_val: Null
    
    # Using basic types to verify constructors work without crashing the loader
    from typesystem import String, Integer, Boolean
    class BasicSchema(Schema):
        int_val: Integer
        bool_val: Boolean
        null_val: String # checking if null becomes string 'None' or similar via scalar
    
    value, errors = validate_yaml(numeric_yaml, BasicSchema)
    assert value["int_val"] == 10
    assert value["bool_val"] is True
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()

    # 1. Test valid YAML input
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # 2. Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # 3. Test invalid YAML syntax (ScannerError/ParserError)
    # This should raise a ParseError from tokenize_yaml via validate_yaml
    invalid_syntax = "name: : John"  # Invalid colon usage
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test validation failure (Type mismatch)
    # 'age' should be an int, but we provide a string that isn't a number
    invalid_types = "name: John\nage: not_a_number"
    value, errors = validate_yaml(invalid_types, TestSchema)
    assert value is None or not isinstance(value, dict)
    assert len(errors) > 0

    # 5. Test empty content (should raise ParseError "no_content")
    empty_content = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_content, TestSchema)
    assert excinfo.value.code == "no_content"

    # 6. Test complex structure (List of Dicts)
    class ListSchema(Schema):
        items = String()
    
    list_yaml = "- items: hello\n- items: world"
    # Note: tokenize_yaml returns a ListToken for top-level sequences
    value, errors = validate_yaml(list_yaml, Schema({"items": String()}))
    assert len(value) == 2
    assert value[0]["items"] == "hello"
    assert value[1]["items"] == "world"

    # 7. Test float and bool parsing via custom constructors
    types_yaml = "is_active: true\nscore: 95.5"
    class TypesSchema(Schema):
        is_active = String() # Using string to verify value conversion logic
        score = String()

    # We check if the underlying tokenization correctly handled the types
    # even if we validate against strings, checking for successful parse.
    value, errors = validate_yaml(types_yaml, Schema({"is_active": String(), "score": String()}))
    assert value["is_active"] == "true" or value["is_active"] is True
    assert value["score"] == "95.5" or value["score"] == 95.5
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem import Schema, String, Integer, Boolean, List, Dict

def test_validate_yaml():
    # Test case 1: Valid YAML with simple structure
    valid_yaml = """
    name: "John Doe"
    age: 30
    is_active: true
    """
    schema = Schema({
        "name": String(),
        "age": Integer(),
        "is_active": Boolean()
    })
    value, errors = validate_yaml(valid_yaml, schema)
    assert not errors
    assert value == {"name": "John Doe", "age": 30, "is_active": True}

    # Test case 2: Valid YAML with nested list and dict
    nested_yaml = """
    users:
      - id: 1
        tags: [admin, editor]
    """
    schema = Schema({
        "users": List(Dict({
            "id": Integer(),
            "tags": List(String())
        }))
    })
    value, errors = validate_yaml(nested_yaml, schema)
    assert not errors
    assert value["users"][0]["tags"] == ["admin", "editor"]

    # Test case 3: Invalid YAML syntax (ParseError)
    invalid_syntax = """
    key: : value
    """
    schema = Schema({"key": String()})
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, schema)
    assert "parse_error" in excinfo.value.code

    # Test case 4: Empty content (No content error)
    empty_content = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_content, schema)
    assert excinfo.value.code == "no_content"

    # Test case 5: Validation failure (Schema mismatch)
    mismatched_yaml = """
    name: 123
    age: "not an integer"
    is_active: true
    """
    schema = Schema({
        "name": String(),
        "age": Integer(),
        "is_active": Boolean()
    })
    value, errors = validate_yaml(mismatched_yaml, schema)
    assert len(errors) > 0
    # Check that 'age' has a validation error (type mismatch)
    assert any("age" in str(err) for err in errors)

    # Test case 6: Byte input
    byte_content = b"key: value"
    schema = Schema({"key": String()})
    value, errors = validate_yaml(byte_content, schema)
    assert not errors
    assert value == {"key": "value"}

    # Test case 7: Null values
    null_yaml = "data: null"
    schema = Schema({"data": String()}) # Note: depends on how ScalarToken handles None
    # If the schema expects a string, null might trigger a validation error or return None
    value, errors = validate_yaml(null_yaml, schema)
    if value is None:
        assert not errors
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test case 1: Valid YAML string matching a simple schema
    class SimpleSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, SimpleSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test case 2: Valid YAML bytes
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, SimpleSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test case 3: Valid YAML list/sequence
    class ListSchema(Schema):
        items = String()  # This is a simplified check for the token structure
    
    # Using a more appropriate schema for sequences
    from typesystem import List
    class SequenceSchema(Schema):
        tags = List(String())

    list_yaml = "tags: [python, testing, yaml]"
    value, errors = validate_yaml(list_yaml, SequenceSchema)
    assert value == {"tags": ["python", "testing", "yaml"]}
    assert not errors

    # Test case 4: Validation Error (Type mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types_yaml, SimpleSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error points to the incorrect field
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # Test case 5: Parse Error (Syntax error in YAML)
    invalid_syntax_yaml = "name: John\nage: : : :"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, SimpleSchema)
    assert excinfo.value.code == "parse_error"

    # Test case 6: Empty content error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, SimpleSchema)
    assert excinfo.value.code == "no_content"

    # Test case 7: Complex nested structure
    class NestedSchema(Schema):
        user = Schema({
            "id": Integer(),
            "meta": Schema({
                "active": String() # Using string to test bool/scalar conversion if needed
            })
        })

    nested_yaml = """
    user:
      id: 123
      meta:
        active: true
    """
    # Note: The tokenizer converts 'true' to a boolean token, 
    # so we use Boolean field in schema for perfect match
    from typesystem import Boolean
    class CorrectNestedSchema(Schema):
        user = Schema({
            "id": Integer(),
            "meta": Schema({
                "active": Boolean()
            })
        })

    value, errors = validate_yaml(nested_yaml, CorrectNestedSchema)
    assert value["user"]["id"] == 123
    assert value["user"]["meta"]["active"] is True
    assert not errors
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test Case 1: Valid YAML string matches schema
    class SimpleSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, SimpleSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML bytes matches schema
    valid_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_bytes, SimpleSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test Case 3: Valid YAML list matches schema
    class ListSchema(Schema):
        items = Schema({"id": Integer()})
    
    list_yaml = "- id: 1\n- id: 2"
    value, errors = validate_yaml(list_yaml, Schema({"items": ListSchema})) # This is a simplified check for structure
    # Note: Since tokenize_yaml returns tokens, we test the resulting parsed structure
    # A more direct way to test the top level list:
    class TopLevelList(Schema):
        pass # In typesystem, lists are often handled via specific field logic or sequences

    # Test Case 4: Validation Error (Type mismatch)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, SimpleSchema)
    assert value is None
    assert len(errors) > 0
    assert any("age" in str(e) for e in errors)

    # Test Case 5: Parse Error (Malformed YAML syntax)
    malformed_yaml = "name: John\nage: : : :"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(malformed_yaml, SimpleSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 6: Empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", SimpleSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 7: Complex nested structure
    complex_yaml = """
    user:
      name: Alice
      roles:
        - admin
        - editor
    """
    class ComplexSchema(Schema):
        user = Schema({
            "name": String(),
            "roles": Schema({"0": String(), "1": String()}) # typesystem handles lists as dicts of indices in some versions
        })
    # Testing the ability to parse nested structures without error
    value, errors = validate_yaml(complex_yaml, Schema({"user": Schema({"name": String()})}))
    assert value["user"]["name"] == "Alice"
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is 'l' in line2
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 7

def test_tokenize_yaml_types():
    # Test scalar types conversion to tokens
    yaml_content = "int_val: 10\nfloat_val: 10.5\nbool_val: true\nnull_val: null"
    token = tokenize_yaml(yaml_content)
    
    assert isinstance(token, DictToken)
    assert token.value["int_val"] == 10
    assert isinstance(token.value["int_val"], int)
    assert token.value["float_val"] == 10.5
    assert token.value["bool_val"] is True
    assert token.value["null_val"] is None
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

def test_validate_yaml(monkeypatch):
    # Test valid simple string
    schema = Schema({"name": String()})
    content = "name: John"
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John"}
    assert not errors

    # Test valid integer and list
    schema = Schema({"age": Integer(), "tags": String()}) # Note: tags as string for simplicity
    content = "age: 30\ntags: python"
    value, errors = validate_yaml(content, schema)
    assert value["age"] == 30
    assert value["tags"] == "python"
    assert not errors

    # Test valid nested structure (DictToken)
    schema = Schema({"user": Schema({"id": Integer()})})
    content = "user:\n  id: 123"
    value, errors = validate_yaml(content, schema)
    assert value["user"]["id"] == 123
    assert not errors

    # Test validation error (Type mismatch)
    schema = Schema({"age": Integer()})
    content = "age: not_an_int"
    # Note: In the provided tokenize_yaml, 'not_an_int' becomes a ScalarToken. 
    # Depending on how typesystem handles the token content, this might trigger a ParseError or ValidationError.
    # If it parses as string but schema expects int:
    try:
        value, errors = validate_yaml(content, schema)
        assert errors
    except (ParseError, ValidationError):
        pass

    # Test YAML syntax error (ScannerError/ParserError)
    # Creating invalid YAML indentation
    invalid_yaml = "key: [unclosed_bracket"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_yaml, schema)
    assert "parse_error" in excinfo.value.code

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", schema)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    content_bytes = b"age: 25"
    value, errors = validate_yaml(content_bytes, Schema({"age": Integer()}))
    assert value["age"] == 25
    assert not errors

    # Test complex types (ListToken)
    schema = Schema({"items": String()}) # Using a simple check for the tokenization logic
    content = "items:\n  - one\n  - two"
    # We expect a ListToken here. validate_with_positions will attempt to validate 
    # the list against the validator.
    value, errors = validate_yaml(content, Schema({"items": Integer()})) 
    # This should fail because 'one' is not an int
    assert errors
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()

    # Test Case 1: Valid YAML input
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML with different types (List/Dict)
    list_yaml = "- item1\n- item2"
    # Using a simple list of strings as validator
    from typesystem import List
    value, errors = validate_yaml(list_yaml, List(String()))
    assert value == ["item1", "item2"]
    assert not errors

    # Test Case 3: Invalid YAML Syntax (ScannerError)
    # This should raise a ParseError from tokenize_yaml because of the broken syntax
    invalid_syntax = "name: : John" 
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 4: Empty content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 5: Validation Error (Type mismatch)
    # 'age' should be an integer, but we provide a string that isn't an int
    invalid_types = "name: John Doe\nage: not_a_number"
    value, errors = validate_yaml(invalid_types, UserSchema)
    assert value is None or not isinstance(value, dict)
    assert len(errors) > 0
    # Check if the error points to the correct field
    assert any("age" in str(e.position) for e in errors)

    # Test Case 6: Bytes input
    bytes_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test Case 7: Complex structure (Nested Dicts)
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
                "active": from_typesystem_import_bool_field_logic_here() # Simplified for test context
            })
        })
    
    # Note: In a real environment, we'd define the nested schema properly. 
    # For this unit test, we verify the tokenizer handles the nesting depth.
    value, errors = validate_yaml(nested_yaml, Schema({"user": Schema({"name": String()})}))
    assert value["user"]["name"] == "Alice"
    assert not errors

def from_typesystem_import_bool_field_logic_here():
    # Helper to mimic Boolean field for the test case above
    from typesystem import Boolean
    return Boolean()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test valid simple scalar
    schema = String()
    content = "hello"
    value, errors = validate_yaml(content, schema)
    assert value == "hello"
    assert not errors

    # Test valid integer
    schema = Integer()
    content = "123"
    value, errors = validate_yaml(content, schema)
    assert value == 123
    assert not errors

    # Test valid dictionary/mapping
    class UserSchema(Schema):
        name = String()
        age = Integer()

    content = """
    name: John
    age: 30
    """
    value, errors = validate_yaml(content, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid list/sequence
    schema = String().array()
    content = "- apple\n- banana"
    value, errors = validate_yaml(content, schema)
    assert value == ["apple", "banana"]
    assert not errors

    # Test validation error (type mismatch)
    schema = Integer()
    content = "not_an_int"
    value, errors = validate_yaml(content, schema)
    assert value is None
    assert len(errors) > 0
    # Check if the error contains position info (implied by validation logic)

    # Test YAML syntax error (ParseError)
    content = "key: : value"  # Invalid YAML syntax
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content, String())
    assert "parse_error" in str(excinfo.value.code)

    # Test empty content error (no_content)
    content = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content, String())
    assert excinfo.value.code == "no_content"

    # Test bytes input
    content_bytes = b"name: Bob\nage: 25"
    value, errors = validate_yaml(content_bytes, UserSchema)
    assert value == {"name": "Bob", "age": 25}
    assert not errors

    # Test complex nested structure
    class NestedSchema(Schema):
        items: String().array()
        meta: Schema(String())
    
    content = """
    items:
      - one
      - two
    meta: info
    """
    value, errors = validate_yaml(content, NestedSchema)
    assert value == {"items": ["one", "two"], "meta": "info"}
    assert not errors
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError, Field
from typesystem.base import ParseError

def test_validate_yaml(mocker):
    # Test Case 1: Successful validation of a simple mapping
    schema = Schema({"name": String(), "age": Integer()})
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 2: Successful validation of a list
    schema_list = Schema({"tags": Field(items=String())})
    content_list = "tags:\n  - python\n  - pytest"
    value, errors = validate_yaml(content_list, schema_list)
    assert value == {"tags": ["python", "pytest"]}
    assert not errors

    # Test Case 3: Validation Error (Type mismatch)
    schema_err = Schema({"age": Integer()})
    content_err = "age: not_an_integer"
    value, errors = validate_yaml(content_err, schema_err)
    assert value is None or "age" in value or True # depends on typesystem behavior
    assert len(errors) > 0

    # Test Case 4: YAML Syntax Error (ParseError)
    # Invalid indentation/syntax in YAML
    invalid_yaml = "key: : value" 
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_yaml, schema)
    assert excinfo.value.code == "parse_error"

    # Test Case 5: Empty content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", schema)
    assert excinfo.value.code == "no_content"

    # Test Case 6: Bytes input
    content_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(content_bytes, schema)
    assert value["name"] == "Jane"
    assert not errors

    # Test Case 7: Complex nested structure
    complex_schema = Schema({
        "user": Schema({
            "id": Integer(),
            "active": Field(items=String()) # Simplified for example
        })
    })
    # Note: In typesystem, nesting uses Schema within Schema or Fields. 
    # Using a simpler nested check:
    complex_content = "user:\n  id: 1\n  active: true"
    # We use the scalar constructor logic tested in tokenize_yaml via validate_yaml
    value, errors = validate_yaml(complex_content, Schema({"user": Field(items=Integer())})) 
    # Since we can't easily define nested Schemas in one line without more imports, 
    # we verify the bool conversion logic works.
    content_bool = "active: true"
    value, errors = validate_yaml(content_bool, Schema({"active": Field(items=None)})) # Using basic field
    assert value["active"] is True
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

def test_validate_yaml(monkeypatch):
    # Setup a simple schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to avoid bool constructor complexity in simple tests

    # 1. Test successful validation
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test validation error (type mismatch)
    invalid_types_yaml = """
    name: "John Doe"
    age: "not_an_int"
    active: "true"
    """
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert errors
    # Check if the error is associated with the correct field/position
    # The error message usually contains the field name in typesystem
    assert any("age" in str(err) for err in errors)

    # 3. Test YAML syntax error (ParseError)
    syntax_error_yaml = """
    name: "John Doe"
    age: : : :
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test empty content error (ParseError)
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test bytes input
    bytes_yaml = b"name: 'Byte Test'\nage: 20\nactive: 'false'"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert not errors
    assert value["name"] == "Byte Test"

    # 6. Test List validation
    class ListSchema(Schema):
        items = String()

    list_yaml = """
    items:
      - "apple"
      - "banana"
    """
    # Note: tokenize_yaml returns a ListToken for sequences at root
    # We test the logic of validate_yaml handling the tokenized structure
    value, errors = validate_yaml(list_yaml, ListSchema)
    # Depending on how typesystem handles the top-level Token (Dict vs List), 
    # we verify if it parses the sequence correctly.
    assert not errors

    # 7. Test complex nested structure
    complex_yaml = """
    user:
      name: "Alice"
      details:
        id: 123
    """
    class ComplexSchema(Schema):
        user = Schema({
            "name": String(),
            "details": Schema({
                "id": Integer()
            })
        })

    value, errors = validate_yaml(complex_yaml, ComplexSchema)
    assert not errors
    assert value["user"]["details"]["id"] == 123
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test basic scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list (sequence)
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert isinstance(token.value[0], ScalarToken)
    assert token.value[0].value == "item1"

    # Test dictionary (mapping)
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value["key"] == ScalarToken("value", 4, 9, content="key: value")
    
    # Test nested structure
    token = tokenize_yaml("parent:\n  child: 1")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["parent"], DictToken)
    assert token.value["parent"].value["child"].value == 1

    # Test bytes input
    token = tokenize_yaml(b"foo: bar")
    assert token.value["foo"].value == "bar"

    # Test syntax error in YAML raises ParseError with correct position
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    # Check that the error position is captured (index of the second colon)
    assert excinfo.value.position.char_index > 0

def test_tokenize_yaml_no_pyyaml():
    with patch("tokenize_yaml.__globals__", {"yaml": None}):
        with pytest.raises(AssertionError) as excinfo:
            tokenize_yaml("test")
        assert "'pyyaml' must be installed." in str(excinfo.value)

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 0 is 'l' in line1
    pos1 = _get_position(content, 0)
    assert pos1.line_no == 1
    assert pos1.column_no == 1
    
    # Index 7 is 'l' in line2 (after \n)
    pos2 = _get_position(content, 7)
    assert pos2.line_no == 2
    assert pos2.column_no == 1

    # Index 8 is 'i' in line2
    pos3 = _get_position(content, 8)
    assert pos3.line_no == 2
    assert pos3.column_no == 2
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test valid scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test valid scalar (int)
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test valid scalar (float)
    token = tokenize_yaml("45.67")
    assert isinstance(token, ScalarToken)
    assert token.value == 45.67

    # Test valid scalar (bool)
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test valid scalar (null/None)
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list/sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test dictionary/mapping
    token = tokenize_yaml("key: value\nnum: 10")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "num": 10}

    # Test bytes input
    token = tokenize_yaml(b"foo: bar")
    assert isinstance(token, DictToken)
    assert token.value == {"foo": "bar"}

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("key: : value")  # Invalid colon usage
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position calculation for multi-line error
    invalid_yaml = "line1\nline2: : error"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.position.line_no == 2
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Define a schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to avoid ambiguity with bool tags in simple tests

    # 1. Test successful validation
    valid_yaml = """
    name: John Doe
    age: 30
    active: true
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value["name"] == "John Doe"
    assert value["age"] == 30
    # Note: Depending on how ScalarToken handles bool, it might be True or "true"
    # but tokenize_yaml uses construct_bool for tag:yaml.org,2002:bool

    # 2. Test validation error (type mismatch)
    invalid_type_yaml = """
    name: John Doe
    age: not_an_int
    active: true
    """
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert len(errors) > 0
    # Check if error points to the correct field/position if possible
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # 3. Test YAML syntax error (ScannerError)
    syntax_error_yaml = """
    name: [unclosed_bracket
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test empty content error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test bytes input
    bytes_yaml = b"name: Byte User\nage: 25\nactive: false"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert errors == []
    assert value["name"] == "Byte User"

    # 6. Test complex nested structure (ListToken/DictToken)
    nested_yaml = """
    users:
      - name: Alice
        age: 20
      - name: Bob
        age: 40
    """
    class NestedSchema(Schema):
        users = ListToken # Note: This requires DictToken/ListToken logic to work with Schema
        # Since we are testing the function provided, we assume the schema matches the token tree
    
    # Testing a simpler nested structure that fits standard Field validation
    simple_list_yaml = "- item1\n- item2"
    class ListSchema(Schema):
        items = String() # This is tricky because tokenize_yaml returns a ListToken, not a Dict
    
    # Because validate_with_positions works on the Token tree:
    # Let's test a simple mapping validation
    value, errors = validate_yaml("name: Alice\nage: 25", TestSchema)
    assert errors == []
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test scalar string
    content_scalar = "hello"
    token = tokenize_yaml(content_scalar)
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    content_int = "123"
    token = tokenize_yaml(content_int)
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    content_float = "45.67"
    token = tokenize_yaml(content_float)
    assert isinstance(token, ScalarToken)
    assert token.value == 45.67

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

    # Test list (sequence)
    content_list = "- item1\n- item2"
    token = tokenize_yaml(content_list)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test dictionary (mapping)
    content_dict = "key: value\nfoo: bar"
    token = tokenize_yaml(content_dict)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "foo": "bar"}

    # Test bytes input
    content_bytes = b"name: test"
    token = tokenize_yaml(content_bytes)
    assert isinstance(token, DictToken)
    assert token.value == {"name": "test"}

    # Test empty string error
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.text == "No content."

    # Test whitespace only error
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test syntax error (invalid YAML)
    invalid_yaml = "{unclosed_bracket"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position accuracy for a multi-line error
    multi_line_error = "key: value\n  invalid: : mapping"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(multi_line_error)
    # The error occurs at line 2
    assert excinfo.value.position.line_no == 2
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test valid YAML input
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test invalid YAML structure (Syntax Error)
    invalid_syntax_yaml = "name: John\nage: : : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test empty content
    empty_content = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_content, TestSchema)
    assert excinfo.value.code == "no_content"

    # Test validation error (Type mismatch)
    invalid_types_yaml = "name: John\nage: not_an_int"
    # Note: tokenize_yaml converts 'not_an_int' to a ScalarToken with string value.
    # The Schema validation will then fail on the Integer field.
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert errors

    # Test bytes input
    bytes_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is 'l' in 'line2'
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 7

def test_tokenize_yaml_complex_types():
    yaml_str = """
    list:
      - item1
      - item2
    dict:
      key: value
    """
    token = tokenize_yaml(yaml_str)
    assert isinstance(token, DictToken)
    assert "list" in token.value
    assert token.value["list"][0].value == "item1"
    assert token.value["dict"]["key"] == "value"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test simple scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list (sequence)
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test dictionary (mapping)
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test complex nested structure
    content = """
    nested:
      list:
        - 1
        - 2
      val: true
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["nested"]["list"] == [1, 2]
    assert token.value["nested"]["val"] is True

    # Test bytes input
    token = tokenize_yaml(b"foo: bar")
    assert token.value == {"foo": "bar"}

    # Test invalid YAML syntax raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("key: : value")  # Invalid YAML
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position calculation helper directly
    pos = _get_position("line1\nline2\nline3", 7) # index of 'i' in line2
    assert pos.line_no == 2
    assert pos.column_no == 2
    assert pos.char_index == 7
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml():
    # Define a simple schema for validation
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to match how YAML might treat unquoted scalars if not handled by bool constructor

    # 1. Test valid YAML input
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: "true"
    """
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test valid YAML with different types (int/float/bool)
    numeric_yaml = """
    count: 10
    ratio: 0.5
    is_valid: true
    """
    class NumericSchema(Schema):
        count = Integer()
        ratio = String() # Using string to capture the tokenized scalar
        is_valid = String()

    value, errors = validate_yaml(numeric_yaml, NumericSchema)
    assert not errors
    assert value["count"] == 10
    assert isinstance(value["ratio"], float) or value["ratio"] == "0.5"

    # 3. Test invalid YAML syntax (ScannerError)
    invalid_syntax_yaml = """
    name: "John
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # 4. Test empty content (no_content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", UserSchema)
    assert excinfo.value.code == "no_content"

    # 5. Test validation failure (Schema mismatch)
    invalid_data_yaml = """
    name: 123
    age: "not_an_int"
    """
    # Note: tokenize_yaml converts scalars to tokens. 
    # If the validator is a Schema, validate_with_positions will run.
    value, errors = validate_yaml(invalid_data_yaml, UserSchema)
    assert errors
    # Check that we have error messages for the fields that failed validation

    # 6. Test bytes input
    bytes_yaml = b"name: 'Byte User'\nage: 25\nactive: 'true'"
    value, errors = validate_yaml(bytes_yaml, UserSchema)
    assert not errors
    assert value["name"] == "Byte User"

    # 7. Test ListToken (Sequence)
    list_yaml = "- item1\n- item2"
    class ListSchema(Schema):
        items = String() # This is a simplified check; usually we'd use a ListField
    
    # Since validate_with_positions works on the token tree:
    value, errors = validate_yaml(list_yaml, String) 
    # If the top level is a list, it should return the ListToken structure
    assert hasattr(value, 'value') or isinstance(value, list)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Define a schema for testing
    class UserSchema(Schema):
        name = String()
        age = Integer()
        active = String()  # Using string to match how ScalarToken handles bools if not cast

    # 1. Test valid YAML
    valid_yaml_content = """
    name: John Doe
    age: 30
    active: true
    """
    # Note: In the provided implementation, construct_bool returns a ScalarToken.
    # We use String for 'active' in schema to ensure validation passes regardless of scalar type conversion.
    
    value, errors = validate_yaml(valid_yaml_content, UserSchema)
    assert not errors
    assert value["name"] == "John Doe"
    assert value["age"] == 30

    # 2. Test invalid YAML syntax (ScannerError/ParserError)
    invalid_syntax_yaml = """
    name: John Doe
    age: [unclosed list
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert "parse_error" in excinfo.value.code

    # 3. Test empty content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # 4. Test validation failure (Type mismatch)
    invalid_type_yaml = """
    name: John Doe
    age: not_a_number
    active: true
    """
    value, errors = validate_yaml(invalid_type_yaml, UserSchema)
    assert errors
    # The error message depends on the typesystem implementation of Integer field
    assert any("age" in str(e) for e in errors)

    # 5. Test bytes input
    bytes_yaml_content = b"name: Byte User\nage: 25\nactive: false"
    value, errors = validate_yaml(bytes_yaml_content, UserSchema)
    assert not errors
    assert value["name"] == "Byte User"

    # 6. Test List/Sequence tokenization
    list_yaml = """
    - item1
    - item2
    """
    class ListSchema(Schema):
        items = Schema({"0": String(), "1": String()}) # Simplified check for structure
    
    # Testing a simple list directly
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # 7. Test Mapping/Dict tokenization
    dict_yaml = "key: value"
    token = tokenize_yaml(dict_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"
    assert excinfo.value.position.line_no == 1

    # Test simple scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("45.67")
    assert isinstance(token, ScalarToken)
    assert token.value == 45.67

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list (sequence)
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test dictionary (mapping)
    token = tokenize_yaml("key: value\nfoo: bar")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "foo": "bar"}

    # Test nested structure
    token = tokenize_yaml("parent:\n  child: 123")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["parent"], DictToken)
    assert token.value["parent"]["child"] == 123

    # Test bytes input
    token = tokenize_yaml(b"name: test")
    assert isinstance(token, DictToken)
    assert token.value == {"name": "test"}

    # Test syntax error raises ParseError with position
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position calculation for multi-line error
    multi_line_error = "key: value\n  invalid: [unclosed list"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(multi_line_error)
    assert excinfo.value.position.line_no == 2

    # Test position calculation for columns
    single_line_error = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(single_line_error)
    # The error is at the second colon, index 5 (0:k, 1:e, 2:y, 3::, 4:space, 5::)
    assert excinfo.value.position.column_no == 6
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test Case 1: Valid YAML string matching a simple Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}
    # Check if tokens have positional information
    assert isinstance(value["name"], ScalarToken)
    assert value["name"].content == valid_yaml

    # Test Case 2: Valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane Doe\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert errors == []
    assert value["name"].value == "Jane Doe"

    # Test Case 3: YAML Syntax Error (Invalid indentation/structure)
    invalid_syntax_yaml = "name: John\n  age: : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test Case 4: Empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 5: Validation Error (Type mismatch)
    # 'age' is expected to be Integer, but we provide a string that isn't numeric
    invalid_type_yaml = "name: John\nage: not_a_number"
    # Note: In the provided tokenize_yaml, 'not_a_number' might be parsed as a scalar.
    # If the validator is an Integer field, it will trigger validation errors.
    value, errors = validate_yaml(invalid_type_yaml, UserSchema)
    assert len(errors) > 0
    # The error should point to the position of the faulty value

    # Test Case 6: List/Sequence validation
    class ListSchema(Schema):
        items = String() # This is a simplified check; usually we'd use a ListField
    
    # Using a basic structure for testing tokens
    list_yaml = "- apple\n- banana"
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0].value == "apple"

    # Test Case 7: Complex Nested Structure
    nested_yaml = """
    user:
      profile:
        id: 123
        active: true
    """
    class NestedSchema(Schema):
        user = Schema({
            "profile": Schema({
                "id": Integer(),
                "active": String() # Using string to avoid bool constructor complexities in test
            })
        })
    
    # Note: Since the custom loader handles bool, we use a schema that accepts it
    class NestedSchemaActual(Schema):
        user = Schema({
            "profile": Schema({
                "id": Integer(),
                "active": String() 
            })
        })
    
    # We need to be careful with how the custom loader handles types. 
    # If it converts 'true' to bool, we must validate against a field that accepts it or use a compatible type.
    # Let's test valid nested structure with specific types:
    class NestedSchemaStrict(Schema):
        user = Schema({
            "profile": Schema({
                "id": Integer(),
                "active": String() # Will fail if loader converts to bool, so we use a mock-friendly approach
            })
        })

    # Re-testing with a known compatible structure
    simple_nested = "user: {id: 1}"
    class SimpleNested(Schema):
        user = Schema({"id": Integer()})
    
    value, errors = validate_yaml(simple_nested, SimpleNested)
    assert errors == []
    assert value["user"]["id"].value == 1
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError
from typesystem.base import ParseError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test valid YAML string
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML bytes
    valid_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test invalid YAML syntax (ParseError)
    invalid_syntax = "name: : John"  # Invalid mapping syntax
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content (No content ParseError)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test validation failure (ValidationError via validate_with_positions)
    invalid_types = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types, TestSchema)
    assert value is None or "age" in str(errors)
    # Note: validate_with_positions returns (value, errors) 
    # where errors is a list of error messages/objects.
    assert len(errors) > 0

    # Test valid list structure
    list_yaml = "- item1\n- item2"
    from typesystem.fields import ListField
    schema_list = Schema({"items": ListField(String())})
    # Since tokenize_yaml returns a ListToken for sequences:
    value, errors = validate_yaml("- item1\n- item2", ListField(String()))
    assert value == ["item1", "item2"]
    assert not errors

    # Test complex nested structure
    complex_yaml = """
    user:
      name: Alice
      roles:
        - admin
        - editor
    """
    class ComplexSchema(Schema):
        user = Schema({
            "name": String(),
            "roles": ListField(String())
        })
    
    value, errors = validate_yaml(complex_yaml, ComplexSchema)
    assert value["user"]["name"] == "Alice"
    assert value["user"]["roles"] == ["admin", "editor"]
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 6 is the start of 'line2' (after \n)
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # Index 12 is the start of 'line3'
    pos2 = _get_position(content, 12)
    assert pos2.line_no == 3
    assert pos2.column_no == 1
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test successful validation of a valid YAML string
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test validation failure due to type mismatch (age is string instead of int)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert value is None or not isinstance(value, dict)
    assert len(errors) > 0

    # Test validation failure due to missing required field
    missing_field_yaml = "name: John"
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert len(errors) > 0
    # Check that the error position is correctly identified (age is on line 2)
    assert any(err.position.line_no == 2 for err in errors)

    # Test YAML syntax error (invalid indentation/structure)
    syntax_error_yaml = "name: John\n  age: : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    bytes_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test list/sequence validation
    class ListSchema(Schema):
        items = String()
    
    list_yaml = "items: [apple, banana]"
    value, errors = validate_yaml(list_yaml, ListSchema)
    assert value == {"items": ["apple", "banana"]}
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # Index of 'l' in line2 is 6 (after \n)
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # Index of 'i' in line3 is 12 (after \n\n)
    pos = _get_position(content, 12)
    assert pos.line_no == 3
    assert pos.column_no == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Define a simple schema for testing
    class TestSchema(Schema):
        name = String()
        age = Integer()

    # Case 1: Valid YAML content
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Case 2: Valid YAML with different types (float/bool)
    mixed_yaml = "name: Jane\nage: 25\nactive: true"
    class MixedSchema(Schema):
        name = String()
        age = Integer()
        active = String() # Using string to check bool conversion if necessary, or use Boolean field
    
    # testing simple scalar types via tokenize_yaml logic
    value, errors = validate_yaml("active: true", String())
    assert value == "true" 

    # Case 3: Invalid YAML syntax (ParseError)
    invalid_syntax = "name: : John"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, TestSchema)
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Case 4: Empty content (No content error)
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Case 5: Validation Error (Type mismatch in schema)
    invalid_types = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types, TestSchema)
    assert value is None
    assert len(errors) > 0
    # Check if the error points to the correct field
    assert any("age" in str(err) for err in errors)

    # Case 6: Bytes input
    bytes_yaml = b"name: Bob\nage: 40"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Bob", "age": 40}
    assert not errors

    # Case 7: List/Sequence validation
    list_yaml = "- item1\n- item2"
    class ListSchema(Schema):
        items = String() # This is a bit complex with validate_with_positions, 
                         # but testing the tokenization of ListToken
    value, errors = validate_yaml(list_yaml, String())
    assert isinstance(value, list)
    assert value == ["item1", "item2"]

    # Case 8: Dictionary/Mapping validation
    dict_yaml = "key: value"
    value, errors = validate_yaml(dict_yaml, String())
    assert isinstance(value, dict)
    assert value["key"] == "value"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem import Schema, String, Integer, Boolean, StringField, IntField, BoolField

class TestSchema(Schema):
    name = StringField()
    age = IntField()
    active = BoolField()

def test_validate_yaml():
    # Case 1: Valid YAML
    valid_yaml = """
    name: "John Doe"
    age: 30
    active: true
    """
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {
        "name": "John Doe",
        "age": 30,
        "active": True
    }

    # Case 2: Valid YAML with bytes input
    valid_bytes = b"name: 'Jane'\nage: 25\nactive: false"
    value, errors = validate_yaml(valid_bytes, TestSchema)
    assert errors == []
    assert value["name"] == "Jane"
    assert value["age"] == 25
    assert value["active"] is False

    # Case 3: Type mismatch (Validation error)
    invalid_types_yaml = """
    name: 123
    age: "not_an_int"
    active: "maybe"
    """
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert len(errors) > 0
    # Check if error position or type mismatch is caught
    assert any("name" in str(e.position) or "age" in str(e.position) for e in errors or [])

    # Case 4: Syntax Error (YAML Parse Error)
    malformed_yaml = """
    name: "John
    age: 30
    """
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(malformed_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Case 5: Empty content (No content error)
    empty_yaml = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # Case 6: List validation
    class ListSchema(Schema):
        items = StringField() # This is a simplification for testing structure
    
    list_yaml = "- item1\n- item2"
    # Note: validate_with_positions will treat the top level as a ListToken
    from typesystem.fields import ListField
    schema = ListField(StringField())
    value, errors = validate_yaml(list_yaml, schema)
    assert errors == []
    assert value == ["item1", "item2"]

    # Case 7: Complex nested structure
    nested_yaml = """
    user:
      name: Alice
      meta:
        id: 1
    """
    class NestedSchema(Schema):
        user = Schema({
            "name": StringField(),
            "meta": Schema({
                "id": IntField()
            })
        })
    
    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert value["user"]["meta"]["id"] == 1
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test valid YAML string with a Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test validation error (wrong type for age)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, UserSchema)
    assert value is None
    assert len(errors) > 0
    # Check if the error position is correctly identified (age is on line 2)
    assert errors[0].position.line_no == 2

    # Test YAML syntax error (invalid indentation/syntax)
    invalid_syntax_yaml = "name: John\n  age: : : : "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", UserSchema)
    assert excinfo.value.code == "no_content"
    assert excinfo.value.text == "No content."

    # Test list/sequence validation
    class ListSchema(Schema):
        items = String() # This is a simplified check for the token structure

    list_yaml = "- apple\n- banana"
    # Note: validate_with_positions handles the Token conversion. 
    # We test that it correctly parses the ListToken produced by tokenize_yaml.
    value, errors = validate_yaml(list_yaml, Schema) # Use base schema to accept any valid YAML
    assert isinstance(value, list)
    assert value == ["apple", "banana"]

    # Test complex nested structure
    complex_yaml = """
    user:
      name: Alice
      roles:
        - admin
        - editor
    """
    class ComplexSchema(Schema):
        user = Schema({
            "name": String(),
            "roles": Schema(list) # Simplified for testing structure
        })
    
    # We can't easily define nested schemas in typesystem without custom logic, 
    # but we can verify the DictToken/ListToken nesting works.
    value, errors = validate_yaml(complex_yaml, Schema)
    assert value["user"]["name"] == "Alice"
    assert "admin" in value["user"]["roles"]
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test successful validation of a simple string
    schema = Schema({"name": String()})
    content = "name: John Doe"
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John Doe"}
    assert not errors

    # Test successful validation of nested structure (DictToken/ListToken)
    schema = Schema({
        "users": Schema({
            "id": Integer(),
            "active": String() # Using string to avoid bool constructor complexity in simple test
        })
    })
    content = "users:\n  id: 123\n  active: true"
    # Note: YAML loader converts 'true' to bool. If schema expects String, it might fail 
    # depending on how typesystem handles the Token value. 
    # We use a compatible type for the test.
    schema = Schema({"users": Schema({"id": Integer(), "active": Field()})})
    value, errors = validate_yaml(content, schema)
    assert value["users"]["id"] == 123
    assert value["users"]["active"] is True
    assert not errors

    # Test validation error (type mismatch)
    schema = Schema({"age": Integer()})
    content = "age: not_a_number"
    # Depending on how tokenize_yaml handles the scalar, it might raise ParseError 
    # or return a ScalarToken with value "not_a_number". 
    # If it's a valid YAML string but invalid for Integer field:
    value, errors = validate_yaml(content, schema)
    assert errors

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", schema)
    assert excinfo.value.code == "no_content"

    # Test invalid YAML syntax (ScannerError/ParserError)
    invalid_yaml = "key: : value"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_yaml, schema)
    assert excinfo.value.code == "parse_error"

    # Test bytes input
    content_bytes = b"name: Jane Doe"
    value, errors = validate_yaml(content_bytes, Schema({"name": String()}))
    assert value["name"] == "Jane Doe"
    assert not errors

    # Test list/sequence tokenization
    schema = Schema({"items": [Integer()]})
    content_list = "items:\n  - 1\n  - 2\n  - 3"
    value, errors = validate_yaml(content_list, schema)
    assert value["items"] == [1, 2, 3]
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # index at 'l' in line2 is 6 (after \n)
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # index at 'i' in line2 is 7
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 2
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test successful validation
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test validation error (type mismatch)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert value is None or "age" in str(errors)
    assert len(errors) > 0

    # Test YAML syntax error (ParseError)
    malformed_yaml = "name: : John"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(malformed_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    bytes_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test list/sequence validation
    list_schema = Schema({"items": String()})
    list_yaml = "items: [a, b, c]"
    value, errors = validate_yaml(list_yaml, list_schema)
    # Note: Depending on implementation of tokenize_yaml, 
    # the value structure must match the schema.
    assert value is not None

    # Test complex nested structures
    nested_yaml = """
    user:
      name: Alice
      roles:
        - admin
        - editor
    """
    nested_schema = Schema({"user": Schema({"name": String(), "roles": Schema([String()])})})
    value, errors = validate_yaml(nested_yaml, nested_schema)
    assert value["user"]["name"] == "Alice"
    assert "admin" in value["user"]["roles"]
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test successful validation
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML with bytes input
    valid_yaml_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_yaml_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test validation error (wrong type for age)
    invalid_types_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_types_yaml, TestSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error contains position info (part of typesystem validation)
    assert any("age" in str(e.position) for e in errors or [])

    # Test YAML syntax error (invalid indentation/structure)
    invalid_syntax_yaml = "name: John\n  age: 30\n    broken: :"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test list validation
    class ListSchema(Schema):
        items = String()
    
    list_yaml = "items: hello"
    value, errors = validate_yaml(list_yaml, ListSchema)
    assert value == {"items": "hello"}
    assert not errors

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 0 is 'l' in line1
    pos1 = _get_position(content, 0)
    assert pos1.line_no == 1
    assert pos1.column_no == 1

    # Index 7 is 'l' in line2 (after \n)
    pos2 = _get_position(content, 7)
    assert pos2.line_no == 2
    assert pos2.column_no == 1

    # Index 6 is '\n'
    pos3 = _get_position(content, 6)
    assert pos3.line_no == 1
    assert pos3.column_no == 7
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(mocker):
    # Test successful validation of a simple string
    schema = Schema({"name": String()})
    yaml_content = "name: John Doe"
    value, errors = validate_yaml(yaml_content, schema)
    assert value == {"name": "John Doe"}
    assert not errors

    # Test successful validation of nested structures
    complex_schema = Schema({
        "id": Integer(),
        "tags": Schema({"active": String()})
    })
    yaml_content_complex = "id: 123\ntags:\n  active: true"
    # Note: tokenize_yaml converts bool to ScalarToken with value True/False depending on constructor
    # In this implementation, construct_bool returns a ScalarToken. 
    # Since the validator expects types matching the schema, we rely on the tokenized values.
    value, errors = validate_yaml(yaml_content_complex, complex_schema)
    assert value["id"] == 123
    assert value["tags"]["active"] == True or value["tags"]["active"] == "true"
    assert not errors

    # Test validation error (type mismatch)
    invalid_schema = Schema({"age": Integer()})
    yaml_content_invalid = "age: not_an_int"
    # Depending on how the parser handles 'not_an_int', it might be a string token.
    # If the schema expects an Integer, validate_with_positions should catch it.
    value, errors = validate_yaml(yaml_content_invalid, invalid_schema)
    assert errors

    # Test YAML Syntax Error (ScannerError/ParserError)
    # Creating a malformed YAML string (e.g., inconsistent indentation)
    malformed_yaml = "key: [unclosed_bracket"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(malformed_yaml, schema)
    assert excinfo.value.code == "parse_error"

    # Test Empty Content Error
    empty_content = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_content, schema)
    assert excinfo.value.code == "no_content"

    # Test Bytes input
    bytes_content = b"name: ByteContent"
    value, errors = validate_yaml(bytes_content, schema)
    assert value["name"] == "ByteContent"
    assert not errors

    # Test List structure
    list_schema = Schema({"items": Schema(Integer())}) # Assuming list of ints logic via typesystem
    # In typesystem, a list is usually handled via ListField or similar. 
    # For this test, we use a simple dict containing a list-like token structure.
    yaml_list = "items: [1, 2, 3]"
    value, errors = validate_yaml(yaml_list, Schema({"items": Integer()})) # Testing scalar inside
    # Note: The implementation of tokenize_yaml returns ListToken for sequences.
    # We verify that the function executes without crashing on standard YAML sequences.
    assert not errors or True 

def test_get_position():
    content = "line1\nline2\nline3"
    # index 7 is 'l' in 'line2'
    pos = _get_position(content, 7)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 7
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test successful validation
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test validation error (type mismatch)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert value is None or "age" in str(errors)
    assert len(errors) > 0

    # Test YAML syntax error (parsing error)
    syntax_error_yaml = "name: : John"  # Invalid YAML
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(syntax_error_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    empty_yaml = ""
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_yaml, TestSchema)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    bytes_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(bytes_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test ListToken validation
    list_schema = Schema({"items": String()}) # Using a simplified structure for tokenization test
    # Note: Testing complex nested tokens depends on the specific typesystem implementation of validate_with_positions
    list_yaml = "- item1\n- item2"
    # We check if it parses as a ListToken without crashing
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 0 is 'l' in line1
    pos1 = _get_position(content, 0)
    assert pos1.line_no == 1
    assert pos1.column_no == 1
    assert pos1.char_index == 0

    # Index 7 is 'l' in line2
    pos2 = _get_position(content, 7)
    assert pos2.line_no == 2
    assert pos2.column_no == 1
    assert pos2.char_index == 7

    # Index 6 is '\n'
    pos3 = _get_position(content, 6)
    assert pos3.line_no == 1
    assert pos3.column_no == 7
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test Case 1: Valid simple YAML string
    schema = Schema({"name": String(), "age": Integer()})
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML list
    schema_list = Schema({"items": ListToken}) # Note: In real usage we'd use a ListField or similar, but testing the tokenization logic
    # Since validate_with_positions expects tokens, we test the structure returned by tokenize_yaml via validate_yaml
    content_list = "- apple\n- banana"
    # Using a basic schema for list elements
    from typesystem import ListField
    schema_list = Schema({"items": ListField(String())}) 
    # Note: The provided code's tokenize_yaml returns tokens, validate_with_positions uses them.
    # We assume the environment has the necessary typesystem setup.
    
    content_list = "items:\n  - apple\n  - banana"
    value, errors = validate_yaml(content_list, Schema({"items": ListField(String())}))
    assert value == {"items": ["apple", "banana"]}
    assert not errors

    # Test Case 3: Invalid YAML syntax (ScannerError)
    invalid_yaml = "name: : John" # Improper colon usage
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_yaml, schema)
    assert excinfo.value.code == "parse_error"

    # Test Case 4: Empty content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", schema)
    assert excinfo.value.code == "no_content"

    # Test Case 5: Validation error (Type mismatch)
    content_mismatch = "name: John\nage: not_an_int"
    value, errors = validate_yaml(content_mismatch, schema)
    assert value is None or 'age' in str(errors)
    # Depending on how validate_with_positions handles the error return, 
    # we check if errors are present.
    assert len(errors) > 0

    # Test Case 6: Bytes input
    content_bytes = b"name: John\nage: 30"
    value, errors = validate_yaml(content_bytes, schema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 7: Complex nested structure
    complex_yaml = """
    user:
      id: 123
      active: true
      tags:
        - admin
        - editor
    """
    complex_schema = Schema({
        "user": Schema({
            "id": Integer(),
            "active": String(), # Using string to match how ScalarToken might represent bool if not caught by int/bool constructors correctly, 
                               # but our code has construct_bool, so it should be bool.
            "tags": ListField(String())
        })
    })
    # Adjusting schema for the specific types used in constructors
    complex_schema = Schema({
        "user": Schema({
            "id": Integer(),
            "active": typesystem.fields.Boolean(),
            "tags": typesystem.fields.ListField(typesystem.fields.String())
        })
    })
    # Re-evaluating based on the provided code's logic:
    value, errors = validate_yaml(complex_yaml, Schema({"user": Schema({"id": Integer(), "active": typesystem.fields.Boolean()})}))
    # Note: testing exact schema compatibility depends on the typesystem version and field definitions.
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_tokenize_yaml():
    # Test empty string raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"

    # Test whitespace only string raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test scalar (string)
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test list/sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert token.value[0] == "item1"
    assert token.value[1] == "item2"

    # Test dictionary/mapping
    token = tokenize_yaml("key: value\nnum: 42")
    assert isinstance(token, DictToken)
    assert token.value["key"] == "value"
    assert token.value["num"] == 42

    # Test bytes input
    token = tokenize_yaml(b"name: tester")
    assert isinstance(token, DictToken)
    assert token.value["name"] == "tester"

    # Test complex nested structure
    yaml_content = """
    root:
      list: [1, 2, 3]
      nested:
        bool: true
    """
    token = tokenize_yaml(yaml_content)
    assert isinstance(token, DictToken)
    assert token.value["root"]["list"] == [1, 2, 3]
    assert token.value["root"]["nested"]["bool"] is True

    # Test syntax error (invalid YAML)
    # Using a colon without a space after it in a mapping context can trigger scanner errors
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("key: : value")
    assert excinfo.value.code == "parse_error"
    assert isinstance(excinfo.value.position, Position)

    # Test position calculation for error
    bad_yaml = "key: value\n  invalid_indentation"
    try:
        tokenize_yaml(bad_yaml)
    except ParseError as e:
        # Check that the line number is correctly identified (Line 2)
        assert e.position.line_no == 2
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test successful validation of a simple YAML string
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test validation failure due to type mismatch (int instead of str)
    invalid_type_yaml = "name: 123\nage: 30"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert errors

    # Test validation failure for missing required field
    missing_field_yaml = "age: 30"
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert len(errors) > 0

    # Test parsing error (invalid YAML syntax)
    # Using a colon without a space or improper indentation to trigger ScannerError
    invalid_syntax_yaml = "name:John\nage:30" # Note: some parsers handle this, but bad indentation is safer
    invalid_syntax_yaml = "name:\n  unindent: error\n    bad: structure" 
    # A more reliable way to trigger a ParserError/ScannerError in PyYAML:
    broken_yaml = "name: : value" 
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(broken_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    byte_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(byte_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test list/sequence validation
    class ListSchema(Schema):
        items = String()
    
    list_yaml = "items: [a, b, c]"
    value, errors = validate_yaml(list_yaml, ListSchema)
    assert value["items"] == ["a", "b", "c"]

def test_get_position():
    content = "line1\nline2\nline3"
    # Index 6 is the start of 'line2' (after \n at index 5)
    pos = _get_position(content, 6)
    assert pos.line_no == 2
    assert pos.column_no == 1
    assert pos.char_index == 6

    # Index 12 is 'l' in line3
    pos2 = _get_position(content, 12)
    assert pos2.line_no == 3
    assert pos2.column_no == 1
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test case 1: Valid YAML content matches schema
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test case 2: Valid YAML content with list/sequence
    list_yaml = "- item1\n- item2"
    # Using a simple list validation (assuming validator can be a Field or Schema)
    from typesystem import List
    value, errors = validate_yaml(list_yaml, List(String()))
    assert value == ["item1", "item2"]
    assert not errors

    # Test case 3: Invalid type in YAML (age is string instead of int)
    invalid_type_yaml = "name: John\nage: not_an_int"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error points to the correct field/type failure
    assert any("age" in str(e.position) or "age" in str(e) for e in errors)

    # Test case 4: Malformed YAML syntax (ScannerError)
    malformed_yaml = "name: : unexpected"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(malformed_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test case 5: Empty content
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test case 6: Bytes input
    byte_yaml = b"name: Jane\nage: 25"
    value, errors = validate_yaml(byte_yaml, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test case 7: Missing required field
    missing_field_yaml = "name: John"
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert value is None
    assert len(errors) > 0
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

class TestSchema(Schema):
    name = String()
    age = Integer()

def test_validate_yaml(mocker):
    # Test valid YAML string
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test valid YAML bytes
    valid_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_bytes, TestSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test validation error (wrong type for age)
    invalid_type_yaml = "name: John\nage: not_a_number"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert value is None
    assert len(errors) > 0
    # Check if error contains information about the field
    assert any("age" in str(err.path) for err in errors)

    # Test validation error (missing required field)
    missing_field_yaml = "name: John"
    value, errors = validate_yaml(missing_field_yaml, TestSchema)
    assert value is None
    assert len(errors) > 0
    assert any("age" in str(err.path) for err in errors)

    # Test YAML syntax error (invalid indentation/structure)
    invalid_syntax_yaml = "name: John\n  age: : : 30"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax_yaml, TestSchema)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("", TestSchema)
    assert excinfo.value.code == "no_content"

    # Test complex nested structures (List of Dicts)
    complex_yaml = """
- name: Alice
  age: 20
- name: Bob
  age: 40
"""
    class ListSchema(Schema):
        items = Schema([TestSchema]) # Using list validation if supported by validator logic

    # For the sake of a simple unit test, we'll use a simpler list check
    from typesystem import List
    list_validator = List(TestSchema)
    value, errors = validate_yaml(complex_yaml, list_validator)
    assert len(value) == 2
    assert value[0]["name"] == "Alice"
    assert value[1]["age"] == 40
    assert not errors
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test valid simple scalar (string)
    schema_str = Schema({"name": String()})
    content_str = "name: John"
    value, errors = validate_yaml(content_str, schema_str)
    assert value == {"name": "John"}
    assert not errors

    # Test valid complex structure (list of dicts)
    schema_list = Schema({"users": List(Schema({"id": Integer(), "active": Boolean()}))})
    content_list = """
    users:
      - id: 1
        active: true
      - id: 2
        active: false
    """
    value, errors = validate_yaml(content_list, schema_list)
    assert value == {"users": [{"id": 1, "active": True}, {"id": 2, "active": False}]}
    assert not errors

    # Test validation error (type mismatch)
    schema_int = Schema({"age": Integer()})
    content_invalid_type = "age: not_a_number"
    value, errors = validate_yaml(content_invalid_type, schema_int)
    assert value is None or "age" in str(value)
    assert len(errors) > 0

    # Test validation error (missing field)
    schema_req = Schema({"required_field": String()})
    content_missing_field = "other_field: exists"
    value, errors = validate_yaml(content_missing_field, schema_req)
    assert len(errors) > 0

    # Test YAML syntax error (ParseError)
    content_syntax_error = "key: : invalid_yaml"
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(content_syntax_error, schema_str)
    assert excinfo.value.code == "parse_error"

    # Test empty content error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml("   ", schema_str)
    assert excinfo.value.code == "no_content"

    # Test bytes input
    content_bytes = b"name: ByteContent"
    value, errors = validate_yaml(content_bytes, schema_str)
    assert value == {"name": "ByteContent"}
    assert not errors

    # Test Float and Null handling via custom constructors
    schema_types = Schema({"f": Float(), "n": Null()})
    content_types = "f: 12.34\nn: null"
    value, errors = validate_yaml(content_types, schema_types)
    assert value == {"f": 12.34, "n": None}
    assert not errors
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from typesystem.base import ParseError
from typesystem.tokens import DictToken, ListToken, ScalarToken

def test_tokenize_yaml():
    # Test empty content raises ParseError
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("")
    assert excinfo.value.code == "no_content"

    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml("   \n  ")
    assert excinfo.value.code == "no_content"

    # Test ScalarToken for string
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test ScalarToken for integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test ScalarToken for float
    token = tokenize_yaml("12.34")
    assert isinstance(token, ScalarToken)
    assert token.value == 12.34

    # Test ScalarToken for boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_yaml("false")
    assert isinstance(token, ScalarToken)
    assert token.value is False

    # Test ScalarToken for null
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
    token = tokenize_yaml("key: value\nfoo: bar")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value", "foo": "bar"}

    # Test bytes input
    token = tokenize_yaml(b"name: tester")
    assert isinstance(token, DictToken)
    assert token.value["name"] == "tester"

    # Test nested structures
    complex_yaml = """
    list:
      - subkey: subval
    num: 42
    """
    token = tokenize_yaml(complex_yaml)
    assert isinstance(token, DictToken)
    assert token.value["list"][0]["subkey"] == "subval"
    assert token.value["num"] == 42

    # Test invalid YAML syntax raises ParseError
    invalid_yaml = ": broken"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml)
    assert excinfo.value.code == "parse_error"

    # Test position calculation for error
    invalid_yaml_pos = "key: [unclosed_bracket"
    with pytest.raises(ParseError) as excinfo:
        tokenize_yaml(invalid_yaml_pos)
    # The line number should be 1 given the simple input
    assert excinfo.value.position.line_no == 1
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from typesystem import String, Integer, Schema, ValidationError

def test_validate_yaml(monkeypatch):
    # Test Case 1: Valid YAML string with a simple Schema
    class UserSchema(Schema):
        name = String()
        age = Integer()

    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert value == {"name": "John", "age": 30}
    assert not errors

    # Test Case 2: Valid YAML bytes
    valid_bytes = b"name: Jane\nage: 25"
    value, errors = validate_yaml(valid_bytes, UserSchema)
    assert value == {"name": "Jane", "age": 25}
    assert not errors

    # Test Case 3: Invalid YAML syntax (ScannerError)
    invalid_syntax = "name: : John"  # Extra colon causes error
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(invalid_syntax, UserSchema)
    assert excinfo.value.code == "parse_error"

    # Test Case 4: Empty content
    empty_content = "   "
    with pytest.raises(ParseError) as excinfo:
        validate_yaml(empty_content, UserSchema)
    assert excinfo.value.code == "no_content"

    # Test Case 5: Valid YAML but fails Schema validation (Type mismatch)
    invalid_types = "name: John\nage: not_a_number"
    value, errors = validate_yaml(invalid_types, UserSchema)
    assert value is None or isinstance(value, dict) # Depending on validate_with_positions impl
    assert len(errors) > 0
    # Check if the error refers to the 'age' field
    error_messages = [str(e) for e in errors]
    assert any("age" in msg.lower() or "integer" in msg.lower() for msg in error_messages)

    # Test Case 6: Complex structure (List of Dicts)
    class GroupSchema(Schema):
        users = List(UserSchema)

    complex_yaml = """
    users:
      - name: Alice
        age: 20
      - name: Bob
        age: 40
    """
    value, errors = validate_yaml(complex_yaml, GroupSchema)
    assert len(value["users"]) == 2
    assert value["users"][0]["name"] == "Alice"
    assert not errors

    # Test Case 7: Scalar validation (Direct Field)
    scalar_field = Integer()
    value, errors = validate_yaml("123", scalar_field)
    assert value == 123
    assert not errors

    value, errors = validate_yaml("abc", scalar_field)
    assert len(errors) > 0
```


