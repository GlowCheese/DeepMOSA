####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class MySchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, MySchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (parse error)
    content = "name: John\nage: thirty"
    value, errors = validate_yaml(content, MySchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test invalid YAML (validation error)
    content = "name: John\nage: -5"
    value, errors = validate_yaml(content, MySchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty content
    content = ""
    try:
        validate_yaml(content, MySchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test bytes input
    content = b"name: John\nage: 30"
    value, errors = validate_yaml(content, MySchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}


# LLM-generated content at query #2
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John
    age: 30
    """

    value, error_messages = validate_yaml(content, ExampleSchema)

    assert value == {"name": "John", "age": 30}
    assert error_messages == []

    invalid_content = """
    name: John
    age: "thirty"
    """

    value, error_messages = validate_yaml(invalid_content, ExampleSchema)

    assert value is None
    assert len(error_messages) == 1
    assert error_messages[0].text == "Must be of type 'int'."
    assert error_messages[0].code == "type_error"
    assert error_messages[0].position.line_no == 3
    assert error_messages[0].position.column_no == 7


# LLM-generated content at query #3
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    valid_yaml = """
    name: John
    age: 30
    """

    invalid_yaml = """
    name: John
    age: 'not an integer'
    """

    empty_yaml = ""

    # Test valid YAML
    result = validate_yaml(valid_yaml, ExampleSchema)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert result[1] is None

    # Test invalid YAML
    result = validate_yaml(invalid_yaml, ExampleSchema)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[1], list)
    assert len(result[1]) > 0

    # Test empty YAML
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    print("All tests passed.")

test_validate_yaml()


# LLM-generated content at query #4
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test case 1: Valid YAML content with a Schema validator
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test case 2: Invalid YAML content with a Field validator
    content = "name: John\nage: 'thirty'"
    result, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'integer'."

    # Test case 3: Empty YAML content
    content = ""
    result, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "No content."

    # Test case 4: Invalid YAML syntax
    content = "name: John\nage: 30\ninvalid"
    result, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "could not find expected ':'."


# LLM-generated content at query #5
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(type="string", max_length=10)
        age = Field(type="integer", minimum=0)

    content = """
    name: "John Doe"
    age: 25
    """
    result, errors = validate_yaml(content, TestSchema)
    assert result == {"name": "John Doe", "age": 25}
    assert errors == []

    invalid_content = """
    name: "This name is way too long"
    age: -5
    """
    result, errors = validate_yaml(invalid_content, TestSchema)
    assert errors != []


# LLM-generated content at query #6
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\nkey2: value2\nkey3:")
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}

    # Test scalar types
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    token = tokenize_yaml("123.456")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.456

    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]


# LLM-generated content at query #7
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = "name: John Doe\nage: 25"
    class Person(Schema):
        name = Field(type="string")
        age = Field(type="integer")
    result = validate_yaml(content, Person)
    assert result[0] == {"name": "John Doe", "age": 25}
    assert result[1] is None

    content = "name: John Doe\nage: twenty"
    result = validate_yaml(content, Person)
    assert result[0] is None
    assert isinstance(result[1], ParseError)


# LLM-generated content at query #8
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    hobbies:
      - Reading
      - Hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John Doe"
    assert token.value["age"] == 30
    assert isinstance(token.value["hobbies"], ListToken)
    assert token.value["hobbies"].value[0] == "Reading"
    assert token.value["hobbies"].value[1] == "Hiking"



# LLM-generated content at query #9
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    is_student: false
    hobbies:
      - reading
      - hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"].value == "John Doe"
    assert token.value["age"].value == 30
    assert token.value["is_student"].value is False
    assert isinstance(token.value["hobbies"], ListToken)
    assert token.value["hobbies"].value[0].value == "reading"
    assert token.value["hobbies"].value[1].value == "hiking"



# LLM-generated content at query #10
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test with empty string
    try:
        tokenize_yaml("")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"
        assert e.position.line_no == 1
        assert e.position.column_no == 1
        assert e.position.char_index == 0

    # Test with invalid YAML
    invalid_yaml = "key: [value"
    try:
        tokenize_yaml(invalid_yaml)
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.code == "parse_error"

    # Test with valid YAML
    valid_yaml = "key: value"
    token = tokenize_yaml(valid_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == len(valid_yaml) - 1
    assert isinstance(token.lookup(["key"]), ScalarToken)
    assert token.lookup(["key"]).value == "value"



# LLM-generated content at query #11
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John Doe\nage: 30"

    result, errors = validate_yaml(yaml_content, ExampleSchema)
    assert errors is None
    assert result == {"name": "John Doe", "age": 30}

    invalid_yaml_content = "name: John Doe\nage: thirty"

    result, errors = validate_yaml(invalid_yaml_content, ExampleSchema)
    assert errors is not None
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'integer'."

    empty_yaml_content = ""
    try:
        validate_yaml(empty_yaml_content, ExampleSchema)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"


# LLM-generated content at query #12
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors is None
    assert result == {"name": "John Doe", "age": 30}

    invalid_content = """
    name: John Doe
    age: thirty
    """
    result, errors = validate_yaml(invalid_content, ExampleSchema)
    assert errors is not None
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'integer'."
    assert errors[0].code == "type_error.integer"


# LLM-generated content at query #13
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    """
    validator = Schema(
        name=Field(type="string"),
        age=Field(type="integer")
    )
    result = validate_yaml(content, validator)
    assert result[0]["name"] == "John Doe"
    assert result[0]["age"] == 30


# LLM-generated content at query #14
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem.fields import Integer, String
    from typesystem.schemas import Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    content = "name: John Doe\nage: 30"
    value, errors = validate_yaml(content, TestSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    content = "name: John Doe\nage: thirty"
    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test invalid YAML (validation error)
    content = "name: John Doe\nage: -5"
    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty YAML
    content = ""
    try:
        validate_yaml(content, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test invalid YAML structure
    content = "- item1\n- item2"
    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"


# LLM-generated content at query #15
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John
    age: 30
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"].value == "John"
    assert token.value["age"].value == 30

    content = """
    - apple
    - banana
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value[0].value == "apple"
    assert token.value[1].value == "banana"

    content = """
    null
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    content = """
    true
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    content = """
    42
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    content = """
    3.14
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    content = ""
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    content = "invalid: yaml: here"
    try:
        tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 16
        assert exc.position.char_index == 15



# LLM-generated content at query #16
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String

    class ExampleSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John Doe\nage: "
    try:
        validate_yaml(invalid_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test invalid data (validation error)
    invalid_data = "name: John Doe\nage: -5"
    value, errors = validate_yaml(invalid_data, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #17
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    result = validate_yaml(valid_yaml, TestSchema)
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (wrong type)
    invalid_yaml = "name: John\nage: thirty"
    try:
        validate_yaml(invalid_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test invalid YAML (missing required field)
    missing_field_yaml = "name: John"
    try:
        validate_yaml(missing_field_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "required"

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #18
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert not errors
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John Doe\nage: thirty"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert errors
    assert "Must be a number." in errors[0].text

    # Test invalid YAML (validation error)
    invalid_yaml = "name: John Doe\nage: 200"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert errors
    assert "Must be less than or equal to 150." in errors[0].text

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #19
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class MySchema(Schema):
        name = Field(type="string", required=True)
        age = Field(type="integer", required=True)

    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, MySchema)
    assert value == {"name": "John", "age": 30}
    assert errors == []

    content = "name: John\nage: thirty"
    value, errors = validate_yaml(content, MySchema)
    assert value is None
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert errors[0].code == "type_error.integer"

    content = "name: John"
    value, errors = validate_yaml(content, MySchema)
    assert value is None
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert errors[0].code == "required_field.missing"

    content = "age: 30"
    value, errors = validate_yaml(content, MySchema)
    assert value is None
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert errors[0].code == "required_field.missing"

    content = ""
    try:
        value, errors = validate_yaml(content, MySchema)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #20
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Valid YAML content
    content = """
    name: John
    age: 30
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Invalid YAML content
    invalid_content = """
    name: John
    age: "thirty"
    """
    value, errors = validate_yaml(invalid_content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Empty YAML content
    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Invalid YAML syntax
    invalid_syntax_content = """
    name: John
    age: 30
    extra_field: "value"
    """
    try:
        validate_yaml(invalid_syntax_content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #21
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class MySchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: John
    age: 30
    """
    result, errors = validate_yaml(yaml_content, MySchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    yaml_content = """
    name: John
    age: '30'
    """
    result, errors = validate_yaml(yaml_content, MySchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error.integer"

    yaml_content = ""
    try:
        validate_yaml(yaml_content, MySchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    yaml_content = """
    name: John
    age: 30
    extra_field: True
    """
    result, errors = validate_yaml(yaml_content, MySchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    yaml_content = "invalid_yaml"
    try:
        validate_yaml(yaml_content, MySchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #22
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Object

    # Test with valid YAML content
    schema = Object(properties={"name": String(), "age": Integer()})
    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, schema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test with invalid YAML content
    content = "name: John\nage: thirty"
    value, errors = validate_yaml(content, schema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."
    assert errors[0].code == "type_error.integer"

    # Test with empty content
    content = ""
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    # Test with invalid YAML syntax
    content = "name: John\n  age: 30"
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.text == "mapping values are not allowed here."
        assert exc.code == "parse_error"


# LLM-generated content at query #23
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"
        assert exc.position.line_no == 2
        assert exc.position.column_no > 0

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test scalar types
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    token = tokenize_yaml("123.456")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.456

    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]

    # Test nested structures
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested": "value"}}


# LLM-generated content at query #24
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    import pytest

    # Test empty content
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml("")
    assert exc_info.value.text == "No content."
    assert exc_info.value.code == "no_content"
    assert exc_info.value.position.line_no == 1
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 0

    # Test invalid YAML content
    invalid_yaml = "key: value\ninvalid_yaml"
    with pytest.raises(ParseError) as exc_info:
        tokenize_yaml(invalid_yaml)
    assert exc_info.value.code == "parse_error"
    assert exc_info.value.position.line_no == 2
    assert exc_info.value.position.column_no == 1
    assert exc_info.value.position.char_index == 11

    # Test valid YAML content
    valid_yaml = "key: value"
    token = tokenize_yaml(valid_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test valid YAML content with nested structure
    nested_yaml = "key:\n  nested_key: value"
    token = tokenize_yaml(nested_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested_key": "value"}}
    assert token.start == 0
    assert token.end == 22

    # Test valid YAML content with list
    list_yaml = "- item1\n- item2"
    token = tokenize_yaml(list_yaml)
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 13

    # Test valid YAML content with scalar types
    scalar_yaml = "key1: 123\nkey2: 45.67\nkey3: true\nkey4: null"
    token = tokenize_yaml(scalar_yaml)
    assert isinstance(token, DictToken)
    assert token.value == {"key1": 123, "key2": 45.67, "key3": True, "key4": None}
    assert token.start == 0
    assert token.end == 45


# LLM-generated content at query #25
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    key1: value1
    key2:
      - item1
      - item2
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key1"], ScalarToken)
    assert isinstance(token.value["key2"], ListToken)
    assert token.value["key1"].value == "value1"
    assert token.value["key2"].value == ["item1", "item2"]



# LLM-generated content at query #26
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors == []
    assert result["name"] == "John"
    assert result["age"] == 30

    invalid_content = "name: John\nage: thirty"
    result, errors = validate_yaml(invalid_content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].text == "Must be of type 'integer'."

    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."


# LLM-generated content at query #27
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int, minimum=0)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert not errors
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (age below minimum)
    invalid_yaml = "name: John Doe\nage: -5"
    value, errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert errors
    assert errors[0]["code"] == "minimum_value"

    # Test invalid YAML (missing required field)
    missing_field_yaml = "name: John Doe"
    value, errors = validate_yaml(missing_field_yaml, ExampleSchema)
    assert errors
    assert errors[0]["code"] == "required"

    # Test invalid YAML (parse error)
    parse_error_yaml = "name: John Doe\nage: thirty"
    value, errors = validate_yaml(parse_error_yaml, ExampleSchema)
    assert errors
    assert errors[0]["code"] == "parse_error"


# LLM-generated content at query #28
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: thirty"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."
    assert errors[0].code == "type_error.integer"

    # Test invalid YAML (validation error)
    invalid_yaml = "name: John\nage: -5"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be greater than or equal to 0."
    assert errors[0].code == "min_value"

    # Test empty YAML
    empty_yaml = ""
    value, errors = validate_yaml(empty_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "No content."
    assert errors[0].code == "no_content"

    # Test with bytes input
    valid_bytes = b"name: Alice\nage: 25"
    value, errors = validate_yaml(valid_bytes, TestSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25}


# LLM-generated content at query #29
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"] == "John Doe"
    assert token.value["age"] == 30

    content = "invalid: 'yaml"
    try:
        tokenize_yaml(content)
    except ParseError as e:
        assert str(e) == "Parse error at line 1, column 10: mapping values are not allowed here."



# LLM-generated content at query #30
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Valid YAML content
    valid_content = "name: John Doe\nage: 30"
    valid_result = validate_yaml(valid_content, ExampleSchema)
    assert valid_result[0]["name"] == "John Doe"
    assert valid_result[0]["age"] == 30
    assert valid_result[1] is None

    # Invalid YAML content (missing required field)
    invalid_content = "name: John Doe"
    invalid_result = validate_yaml(invalid_content, ExampleSchema)
    assert invalid_result[1] is not None

    # Invalid YAML content (wrong data type)
    invalid_type_content = "name: John Doe\nage: 'thirty'"
    invalid_type_result = validate_yaml(invalid_type_content, ExampleSchema)
    assert invalid_type_result[1] is not None

    # Empty YAML content
    empty_content = ""
    empty_result = validate_yaml(empty_content, ExampleSchema)
    assert empty_result[1] is not None

    # Invalid YAML format
    invalid_format_content = "name: John Doe\nage: 30\n}"
    invalid_format_result = validate_yaml(invalid_format_content, ExampleSchema)
    assert invalid_format_result[1] is not None


# LLM-generated content at query #31
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    test_content = """
    name: John
    age: 30
    """
    result, errors = validate_yaml(test_content, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    test_content_invalid = """
    name: John
    age: "not_an_integer"
    """
    result, errors = validate_yaml(test_content_invalid, TestSchema)
    assert errors is not None
    assert len(errors) == 1
    assert errors[0].code == "type_error.integer"

    test_content_empty = ""
    try:
        validate_yaml(test_content_empty, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    test_content_invalid_yaml = "name: John\nage: 30\ninvalid_yaml: "
    try:
        validate_yaml(test_content_invalid_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #32
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"
        assert exc.position.line_no == 2
        assert exc.position.column_no == 1
        assert exc.position.char_index == 10

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    assert token.content == "key: value"

    # Test scalar types
    token = tokenize_yaml("42")
    assert isinstance(token, ScalarToken)
    assert token.value == 42
    assert token.start == 0
    assert token.end == 1
    assert token.content == "42"

    token = tokenize_yaml("3.14")
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14
    assert token.start == 0
    assert token.end == 3
    assert token.content == "3.14"

    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3
    assert token.content == "true"

    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3
    assert token.content == "null"

    # Test sequence
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 13
    assert token.content == "- item1\n- item2"

    # Test mapping
    token = tokenize_yaml("key1: value1\nkey2: value2")
    assert isinstance(token, DictToken)
    assert token.value == {"key1": "value1", "key2": "value2"}
    assert token.start == 0
    assert token.end == 23
    assert token.content == "key1: value1\nkey2: value2"


# LLM-generated content at query #33
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Object

    class ExampleSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John Doe\nage: thirty"
    try:
        validate_yaml(invalid_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test validation error (age out of bounds)
    invalid_value_yaml = "name: John Doe\nage: 200"
    value, errors = validate_yaml(invalid_value_yaml, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "maximum"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #34
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Valid YAML input
    valid_yaml = "name: John Doe\nage: 30"
    result, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == [], "Expected no errors for valid YAML input"

    # Invalid YAML input (missing required field)
    invalid_yaml = "name: John Doe"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert errors, "Expected errors for invalid YAML input"

    # Invalid YAML input (incorrect type)
    invalid_yaml_type = "name: John Doe\nage: thirty"
    result, errors = validate_yaml(invalid_yaml_type, TestSchema)
    assert errors, "Expected errors for YAML input with incorrect type"

    # Empty YAML input
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.text == "No content.", "Expected 'No content.' error for empty YAML input"


# LLM-generated content at query #35
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    result = validate_yaml(valid_yaml, TestSchema)
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (wrong type)
    invalid_yaml = "name: John\nage: thirty"
    try:
        validate_yaml(invalid_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "validation_error"
        assert "Must be of type 'int'." in exc.text

    # Test invalid YAML (missing field)
    missing_field_yaml = "name: John"
    try:
        validate_yaml(missing_field_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "validation_error"
        assert "The field 'age' is required." in exc.text

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    print("All tests passed!")

if __name__ == "__main__":
    test_validate_yaml()


# LLM-generated content at query #36
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class UserSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, UserSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    content = "name: John\nage: 'thirty'"
    result, errors = validate_yaml(content, UserSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error.integer"

    content = ""
    try:
        validate_yaml(content, UserSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
    else:
        assert False, "Expected ParseError"

    content = "name: John\nage: 30\ninvalid: field"
    result, errors = validate_yaml(content, UserSchema)
    assert len(errors) == 1
    assert errors[0].code == "unknown_field"


# LLM-generated content at query #37
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: '30'
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].text == "Must be of type 'int'."

    content = """
    name: John Doe
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value is None
    assert len(errors) == 1
    assert errors[0].code == "required"
    assert errors[0].text == "This field is required."

    content = """
    name: John Doe
    age: 30
    extra: field
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
    - field1
    - field2
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
      nested:
        field: value
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
      nested:
        field: value
        invalid: field
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
      nested:
        field: value
        invalid:
          field: value
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
      nested:
        field: value
        invalid:
          field: value
          nested:
            field: value
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    content = """
    name: John Doe
    age: 30
    extra:
      field: value
      nested:
        field: value
        invalid:
          field: value
          nested:
            field: value
            invalid: field
    """
    value, errors = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert errors == []


# LLM-generated content at query #38
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    result = validate_yaml(valid_yaml, TestSchema)
    assert result[0] == {"name": "John", "age": 30}
    assert result[1] == []

    # Test invalid YAML (missing required field)
    invalid_yaml = "name: John"
    result = validate_yaml(invalid_yaml, TestSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].text == "The field 'age' is required."

    # Test invalid YAML (wrong type)
    invalid_type_yaml = "name: John\nage: thirty"
    result = validate_yaml(invalid_type_yaml, TestSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert "Must be an integer." in result[1][0].text

    # Test empty YAML
    empty_yaml = ""
    result = validate_yaml(empty_yaml, TestSchema)
    assert result[0] is None
    assert len(result[1]) == 1
    assert result[1][0].text == "No content."

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #39
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Object

    class ExampleSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    content = "name: John Doe\nage: 30"
    value, errors = validate_yaml(content, ExampleSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_content = "name: John Doe\nage: thirty"
    try:
        value, errors = validate_yaml(invalid_content, ExampleSchema)
    except ParseError as exc:
        assert "parse_error" in exc.code
    else:
        assert False, "Expected ParseError"

    # Test invalid data (validation error)
    invalid_data = "name: A very long name that exceeds the maximum length\nage: 30"
    value, errors = validate_yaml(invalid_data, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "max_length"

    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
    else:
        assert False, "Expected ParseError"

    print("All test cases pass")

test_validate_yaml()


# LLM-generated content at query #40
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test empty content
    try:
        validate_yaml("", Field(type="string"))
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    # Test invalid YAML
    try:
        validate_yaml("key: [1, 2", Field(type="object"))
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test valid YAML with validation error
    class ExampleSchema(Schema):
        name = Field(type="string", max_length=5)

    value, errors = validate_yaml("name: too_long_name", ExampleSchema)
    assert errors is not None
    assert errors[0].text == "Must have no more than 5 characters."

    # Test valid YAML with successful validation
    value, errors = validate_yaml("name: short", ExampleSchema)
    assert value == {"name": "short"}
    assert errors is None


# LLM-generated content at query #41
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test case 1: Valid YAML with a simple schema
    class SimpleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = "name: John\nage: 30"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors == [], "Expected no errors for valid YAML"

    # Test case 2: Invalid YAML with a simple schema
    yaml_content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(yaml_content, SimpleSchema)
    assert errors, "Expected errors for invalid YAML"

    # Test case 3: Empty YAML content
    yaml_content = ""
    try:
        value, errors = validate_yaml(yaml_content, SimpleSchema)
    except ParseError as e:
        assert e.text == "No content.", "Expected 'No content.' error for empty YAML"

    # Test case 4: Invalid YAML syntax
    yaml_content = "name: John\nage: 30\ninvalid:"
    try:
        value, errors = validate_yaml(yaml_content, SimpleSchema)
    except ParseError as e:
        assert e.code == "parse_error", "Expected parse error for invalid YAML syntax"


# LLM-generated content at query #42
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.code == "parse_error"
        assert exc.position.line_no == 2
        assert exc.position.column_no == 1
        assert exc.position.char_index == 10

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9

    # Test scalar tokens
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2

    # Test sequence tokens
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 13


# LLM-generated content at query #43
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(type="string")
        age = Field(type="integer")

    content = "name: John Doe\nage: 30"
    result = validate_yaml(content, ExampleSchema)
    assert result[0] == {"name": "John Doe", "age": 30}
    assert result[1] == []

    content = "name: John Doe\nage: thirty"
    result = validate_yaml(content, ExampleSchema)
    assert result[0] == {"name": "John Doe", "age": "thirty"}
    assert len(result[1]) == 1
    assert result[1][0].code == "type_error"

    content = "name: John Doe\nage: 30\ninvalid: true"
    result = validate_yaml(content, ExampleSchema)
    assert result[0] == {"name": "John Doe", "age": 30, "invalid": True}
    assert len(result[1]) == 1
    assert result[1][0].code == "invalid_key"

    content = "name: John Doe"
    result = validate_yaml(content, ExampleSchema)
    assert result[0] == {"name": "John Doe"}
    assert len(result[1]) == 1
    assert result[1][0].code == "required"


# LLM-generated content at query #44
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    schema = Schema.of({"name": Field(type="string", required=True)})
    # Test valid YAML
    content = "name: John"
    value, errors = validate_yaml(content, schema)
    assert value == {"name": "John"}
    assert not errors

    # Test invalid YAML (missing required field)
    content = "age: 30"
    value, errors = validate_yaml(content, schema)
    assert errors == [{'code': 'required', 'text': 'The field "name" is required.', 'position': Position(line_no=1, column_no=1, char_index=0)}]

    # Test invalid YAML (syntax error)
    content = "name: John:"
    try:
        validate_yaml(content, schema)
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.position.line_no == 1
        assert exc.position.column_no == 10


# LLM-generated content at query #45
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String

    class ExampleSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_content = "name: John\nage: 30"
    value, errors = validate_yaml(valid_content, ExampleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (parse error)
    invalid_content = "name: John\nage: thirty"
    try:
        value, errors = validate_yaml(invalid_content, ExampleSchema)
    except ParseError as exc:
        assert "could not determine a constructor for the tag 'tag:yaml.org,2002:thirty'" in str(exc)

    # Test invalid YAML (validation error)
    invalid_content = "name: John\nage: 200"
    value, errors = validate_yaml(invalid_content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be less than or equal to 150."

    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.text == "No content."

    # Test bytes input
    bytes_content = b"name: Alice\nage: 25"
    value, errors = validate_yaml(bytes_content, ExampleSchema)
    assert errors == []
    assert value == {"name": "Alice", "age": 25}


# LLM-generated content at query #46
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    result, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: thirty"
    try:
        validate_yaml(invalid_yaml, TestSchema)
    except ParseError as exc:
        assert "could not found expected ':'" in str(exc)

    # Test validation error
    invalid_data = "name: John\nage: -5"
    result, errors = validate_yaml(invalid_data, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be greater than or equal to 0."
    assert errors[0].code == "minimum"

    # Test empty content
    try:
        validate_yaml("", TestSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    print("All tests passed!")

if __name__ == "__main__":
    test_validate_yaml()


# LLM-generated content at query #47
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Object

    class PersonSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    yaml_content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = """
    name: John Doe
    age: thirty
    """
    try:
        validate_yaml(invalid_yaml, PersonSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test validation error
    invalid_content = """
    name: John Doe
    age: 200
    """
    value, errors = validate_yaml(invalid_content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "maximum"

    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, PersonSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test bytes input
    bytes_content = b"""
    name: Jane Doe
    age: 25
    """
    value, errors = validate_yaml(bytes_content, PersonSchema)
    assert errors == []
    assert value == {"name": "Jane Doe", "age": 25}


# LLM-generated content at query #48
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    hobbies:
      - Reading
      - Hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value['name'], ScalarToken)
    assert token.value['name'].value == 'John Doe'
    assert isinstance(token.value['age'], ScalarToken)
    assert token.value['age'].value == 30
    assert isinstance(token.value['hobbies'], ListToken)
    assert isinstance(token.value['hobbies'].value[0], ScalarToken)
    assert token.value['hobbies'].value[0].value == 'Reading'
    assert isinstance(token.value['hobbies'].value[1], ScalarToken)
    assert token.value['hobbies'].value[1].value == 'Hiking'

    # Test empty content
    try:
        tokenize_yaml("")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"

    # Test invalid YAML
    invalid_content = """
    name: John Doe
    age: 30
    hobbies:
      - Reading
      - Hiking
    invalid: True
    """
    try:
        tokenize_yaml(invalid_content)
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #49
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = "name: John\nage: 30"
    validator = Schema(
        name=Field(type="string"),
        age=Field(type="integer"),
    )
    value, error_messages = validate_yaml(content, validator)
    assert error_messages == []
    assert value == {"name": "John", "age": 30}

    content = "name: John\nage: thirty"
    validator = Schema(
        name=Field(type="string"),
        age=Field(type="integer"),
    )
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) == 1
    assert error_messages[0].text == "Must be a number."
    assert value is None

    content = "name: John\nage:"
    validator = Schema(
        name=Field(type="string"),
        age=Field(type="integer"),
    )
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) == 1
    assert error_messages[0].text == "This field is required."
    assert value is None

    content = ""
    validator = Schema(
        name=Field(type="string"),
        age=Field(type="integer"),
    )
    value, error_messages = validate_yaml(content, validator)
    assert len(error_messages) == 1
    assert error_messages[0].text == "No content."
    assert value is None


# LLM-generated content at query #50
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    is_active: true
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.value["name"].value == "John Doe"
    assert token.value["age"].value == 30
    assert token.value["is_active"].value is True

    content = """
    - apple
    - banana
    - cherry
    """
    token = tokenize_yaml(content)
    assert isinstance(token, ListToken)
    assert token.value[0].value == "apple"
    assert token.value[1].value == "banana"
    assert token.value[2].value == "cherry"

    content = "42"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 42

    content = "3.14"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value == 3.14

    content = "true"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is True

    content = "null"
    token = tokenize_yaml(content)
    assert isinstance(token, ScalarToken)
    assert token.value is None

    content = ""
    try:
        token = tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    content = "invalid: yaml: here"
    try:
        token = tokenize_yaml(content)
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"


# LLM-generated content at query #51
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    hobby:
      - Reading
      - Running
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token.start == 5
    assert token.end == len(content) - 1
    assert isinstance(token.value['name'], ScalarToken)
    assert token.value['name'].value == 'John Doe'
    assert isinstance(token.value['age'], ScalarToken)
    assert token.value['age'].value == 30
    assert isinstance(token.value['hobby'], ListToken)
    assert isinstance(token.value['hobby'].value[0], ScalarToken)
    assert token.value['hobby'].value[0].value == 'Reading'
    assert isinstance(token.value['hobby'].value[1], ScalarToken)
    assert token.value['hobby'].value[1].value == 'Running'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)

    content = """
    name: "test"
    """

    value, errors = validate_yaml(content, TestSchema)
    assert errors == []
    assert value == {"name": "test"}

    content = """
    name: 123
    """

    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    content = ""
    try:
        value, errors = validate_yaml(content, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
    else:
        assert False, "Expected ParseError"


# LLM-generated content at query #2
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """

    value, error_messages = validate_yaml(content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert error_messages == []

    invalid_content = """
    name: John Doe
    age: "thirty"
    """

    value, error_messages = validate_yaml(invalid_content, ExampleSchema)
    assert value == {"name": "John Doe", "age": "thirty"}
    assert len(error_messages) == 1
    assert error_messages[0].text == "Must be of type 'integer'."

    empty_content = ""
    try:
        validate_yaml(empty_content, ExampleSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"


# LLM-generated content at query #3
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema

    class PersonSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(valid_yaml, PersonSchema)
    assert errors == []
    assert result == {"name": "John Doe", "age": 30}

    # Test invalid YAML (age out of range)
    invalid_yaml = """
    name: John Doe
    age: 200
    """
    result, errors = validate_yaml(invalid_yaml, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "maximum"

    # Test invalid YAML (missing required field)
    missing_field_yaml = """
    name: John Doe
    """
    result, errors = validate_yaml(missing_field_yaml, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test invalid YAML (invalid syntax)
    invalid_syntax_yaml = """
    name: John Doe
    age: thirty
    """
    result, errors = validate_yaml(invalid_syntax_yaml, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "parse_error"

    # Test empty YAML
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "no_content"

    print("All tests passed!")

if __name__ == "__main__":
    test_validate_yaml()


# LLM-generated content at query #4
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Object

    class PersonSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    yaml_content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = """
    name: John Doe
    age: thirty
    """
    value, errors = validate_yaml(invalid_yaml, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, PersonSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test YAML with validation error
    invalid_content = """
    name: John Doe
    age: -5
    """
    value, errors = validate_yaml(invalid_content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #5
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"
        assert exc.position.line_no == 2
        assert exc.position.column_no >= 1

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], ScalarToken)
    assert token.value["key"].value == "value"

    # Test nested YAML
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["key"], DictToken)
    assert isinstance(token.value["key"].value["nested"], ScalarToken)
    assert token.value["key"].value["nested"].value == "value"

    # Test list YAML
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert all(isinstance(item, ScalarToken) for item in token.value)
    assert token.value[0].value == "item1"
    assert token.value[1].value == "item2"

    # Test scalar types
    token = tokenize_yaml("int: 42\nfloat: 3.14\nbool: true\nnull: null")
    assert isinstance(token, DictToken)
    assert isinstance(token.value["int"], ScalarToken)
    assert token.value["int"].value == 42
    assert isinstance(token.value["float"], ScalarToken)
    assert token.value["float"].value == 3.14
    assert isinstance(token.value["bool"], ScalarToken)
    assert token.value["bool"].value is True
    assert isinstance(token.value["null"], ScalarToken)
    assert token.value["null"].value is None


# LLM-generated content at query #6
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John Doe\nage: thirty"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_type"

    # Test invalid YAML (validation error)
    invalid_yaml = "name: John Doe\nage: -5"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test YAML with extra fields (should be ignored by default)
    extra_fields_yaml = "name: John Doe\nage: 30\noccupation: Developer"
    value, errors = validate_yaml(extra_fields_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #7
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (missing required field)
    invalid_yaml = "name: John"
    value, errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].text == "The field 'age' is required."

    # Test invalid YAML (wrong type)
    invalid_type_yaml = "name: John\nage: thirty"
    value, errors = validate_yaml(invalid_type_yaml, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'int'."

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    # Test invalid YAML syntax
    invalid_syntax_yaml = "name: John\n age: 30"  # Extra space before age
    try:
        validate_yaml(invalid_syntax_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #8
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML content
    valid_content = "name: John\nage: 30"
    value, errors = validate_yaml(valid_content, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML content (parse error)
    invalid_content = "name: John\nage: thirty"
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."
    assert errors[0].code == "type_error.number"

    # Test invalid YAML content (validation error)
    invalid_content = "name: John\nage: 200"
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be less than or equal to 150."
    assert errors[0].code == "max_value"

    # Test empty YAML content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    # Test invalid YAML syntax
    invalid_syntax = "name: John\n  age: 30"
    try:
        validate_yaml(invalid_syntax, TestSchema)
    except ParseError as exc:
        assert exc.text == "mapping values are not allowed here."
        assert exc.code == "parse_error"


# LLM-generated content at query #9
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test case 1: Empty string
    try:
        tokenize_yaml("")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert e.text == "No content.", f"Expected 'No content.', got {e.text}"

    # Test case 2: Valid YAML string
    result = tokenize_yaml("key: value")
    assert isinstance(result, DictToken), "Expected DictToken"
    assert result.value == {"key": "value"}, f"Expected {{'key': 'value'}}, got {result.value}"

    # Test case 3: Invalid YAML string
    try:
        tokenize_yaml("key: value\nkey2:")
        assert False, "Expected ParseError"
    except ParseError as e:
        assert "could not find expected ':'" in e.text, f"Expected error about missing ':', got {e.text}"



# LLM-generated content at query #10
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML content
    valid_yaml = "name: John Doe\nage: 30"
    value, error_messages = validate_yaml(valid_yaml, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert error_messages == []

    # Test invalid YAML content (missing required field)
    invalid_yaml = "name: John Doe"
    value, error_messages = validate_yaml(invalid_yaml, ExampleSchema)
    assert value == {"name": "John Doe"}
    assert len(error_messages) == 1
    assert error_messages[0].code == "required"

    # Test invalid YAML content (invalid field type)
    invalid_yaml = "name: John Doe\nage: thirty"
    value, error_messages = validate_yaml(invalid_yaml, ExampleSchema)
    assert value == {"name": "John Doe", "age": "thirty"}
    assert len(error_messages) == 1
    assert error_messages[0].code == "type_error"

    # Test invalid YAML content (invalid YAML syntax)
    invalid_yaml = "name: John Doe\nage: thirty"
    try:
        validate_yaml(invalid_yaml, ExampleSchema)
    except ParseError as e:
        assert e.code == "parse_error"

    # Test empty YAML content
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as e:
        assert e.code == "no_content"


# LLM-generated content at query #11
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors is None
    assert result == {"name": "John Doe", "age": 30}

    content = """
    name: John Doe
    age: 'thirty'
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors is not None
    assert len(errors) == 1
    assert errors[0].code == "type_error"


# LLM-generated content at query #12
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """
    result, error_messages = validate_yaml(content, ExampleSchema)
    assert error_messages == []
    assert result == {"name": "John Doe", "age": 30}

    content = """
    name: John Doe
    age: 'thirty'
    """
    result, error_messages = validate_yaml(content, ExampleSchema)
    assert len(error_messages) == 1
    assert error_messages[0].text == "Must be of type 'integer'."

    content = ""
    result, error_messages = validate_yaml(content, ExampleSchema)
    assert len(error_messages) == 1
    assert error_messages[0].text == "No content."

    content = """
    name: John Doe
    """
    result, error_messages = validate_yaml(content, ExampleSchema)
    assert len(error_messages) == 1
    assert error_messages[0].text == "The field 'age' is required."


# LLM-generated content at query #13
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: "John"
    age: 30
    """
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert not errors
    assert value == {"name": "John", "age": 30}

    yaml_content = """
    name: "John"
    age: "thirty"
    """
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert errors
    assert str(errors[0]) == "Must be of type 'int'."

    yaml_content = """
    name: "John"
    """
    value, errors = validate_yaml(yaml_content, PersonSchema)
    assert errors
    assert str(errors[0]) == "This field is required."

    yaml_content = ""
    try:
        validate_yaml(yaml_content, PersonSchema)
    except ParseError as exc:
        assert str(exc) == "No content."


# LLM-generated content at query #14
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    is_active: true
    hobbies:
      - reading
      - hiking
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert token["name"].value == "John Doe"
    assert token["age"].value == 30
    assert token["is_active"].value is True
    assert isinstance(token["hobbies"], ListToken)
    assert token["hobbies"][0].value == "reading"
    assert token["hobbies"][1].value == "hiking"


# LLM-generated content at query #15
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    """

    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)

    result, errors = validate_yaml(content, PersonSchema)
    assert result == {"name": "John Doe", "age": 30}
    assert errors == []


# LLM-generated content at query #16
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    """

    class UserSchema(Schema):
        name = Field(str)
        age = Field(int)

    value, errors = validate_yaml(content, UserSchema)

    assert value == {"name": "John Doe", "age": 30}
    assert errors == []

    invalid_content = """
    name: John Doe
    age: 'thirty'
    """

    value, errors = validate_yaml(invalid_content, UserSchema)

    assert value is None
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'int'."
    assert errors[0].code == "type_error"


# LLM-generated content at query #17
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty string
    try:
        tokenize_yaml("")
        assert False, "Expected ParseError for empty string"
    except ParseError as e:
        assert e.code == "no_content"

    # Test simple scalar
    token = tokenize_yaml("hello")
    assert isinstance(token, ScalarToken)
    assert token.value == "hello"

    # Test integer
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123

    # Test float
    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45

    # Test boolean
    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True

    # Test null
    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None

    # Test simple list
    token = tokenize_yaml("- hello\n- world")
    assert isinstance(token, ListToken)
    assert len(token.value) == 2
    assert isinstance(token.value[0], ScalarToken)
    assert token.value[0].value == "hello"
    assert isinstance(token.value[1], ScalarToken)
    assert token.value[1].value == "world"

    # Test simple map
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert isinstance(token.value["key"], ScalarToken)
    assert token.value["key"].value == "value"

    # Test nested structures
    token = tokenize_yaml("key:\n  nested_key: nested_value")
    assert isinstance(token, DictToken)
    assert len(token.value) == 1
    assert isinstance(token.value["key"], DictToken)
    assert len(token.value["key"].value) == 1
    assert isinstance(token.value["key"].value["nested_key"], ScalarToken)
    assert token.value["key"].value["nested_key"].value == "nested_value"

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\n  invalid_indent: value")
        assert False, "Expected ParseError for invalid YAML"
    except ParseError as e:
        assert e.code == "parse_error"



# LLM-generated content at query #18
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(type="string")
        age = Field(type="integer")

    # Test valid YAML
    valid_yaml = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (missing required field)
    invalid_yaml = """
    name: John Doe
    """
    value, errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test invalid YAML (incorrect type)
    invalid_yaml_type = """
    name: John Doe
    age: thirty
    """
    value, errors = validate_yaml(invalid_yaml_type, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test invalid YAML syntax
    invalid_syntax_yaml = """
    name: John Doe
    age: 30
    extra_field: 
    """
    try:
        validate_yaml(invalid_syntax_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #19
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    schema = Schema({"name": Field(str), "age": Field(int)})
    content = "name: John\nage: 30"
    result, errors = validate_yaml(content, schema)
    assert errors is None
    assert result == {"name": "John", "age": 30}

    content = "name: John\nage: thirty"
    result, errors = validate_yaml(content, schema)
    assert errors is not None
    assert errors[0].text == "Must be of type 'integer'."

    content = "name: John\nage:"
    result, errors = validate_yaml(content, schema)
    assert errors is not None
    assert errors[0].text == "This field is required."

    content = ""
    result, errors = validate_yaml(content, schema)
    assert errors is not None
    assert errors[0].text == "No content."


# LLM-generated content at query #20
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Valid YAML
    content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors == []
    assert result == {'name': 'John Doe', 'age': 30}

    # Invalid YAML (missing required field)
    content = """
    name: John Doe
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == 'required'

    # Invalid YAML (incorrect type)
    content = """
    name: John Doe
    age: 'thirty'
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == 'type'


# LLM-generated content at query #21
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (missing required field)
    invalid_yaml = "name: John"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "required"

    # Test invalid YAML (wrong type)
    invalid_type_yaml = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "type"

    # Test invalid YAML (parse error)
    parse_error_yaml = "name: John\nage: 30\ninvalid"
    try:
        validate_yaml(parse_error_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #22
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\ninvalid")
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    assert token.content == "key: value"

    # Test nested YAML
    token = tokenize_yaml("key:\n  nested: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": {"nested": "value"}}
    assert token.start == 0
    assert token.end == 17
    assert token.content == "key:\n  nested: value"

    # Test list YAML
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 13
    assert token.content == "- item1\n- item2"

    # Test scalar types
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2
    assert token.content == "123"

    token = tokenize_yaml("123.45")
    assert isinstance(token, ScalarToken)
    assert token.value == 123.45
    assert token.start == 0
    assert token.end == 5
    assert token.content == "123.45"

    token = tokenize_yaml("true")
    assert isinstance(token, ScalarToken)
    assert token.value is True
    assert token.start == 0
    assert token.end == 3
    assert token.content == "true"

    token = tokenize_yaml("null")
    assert isinstance(token, ScalarToken)
    assert token.value is None
    assert token.start == 0
    assert token.end == 3
    assert token.content == "null"

    # Test bytes input
    token = tokenize_yaml(b"key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    assert token.content == "key: value"


# LLM-generated content at query #23
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    """
    class Person(Schema):
        name = Field(str)
        age = Field(int)
    result, errors = validate_yaml(content, Person)
    assert errors == []
    assert result == {"name": "John Doe", "age": 30}


# LLM-generated content at query #24
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = """
    name: John Doe
    age: 30
    """
    result, errors = validate_yaml(content, ExampleSchema)
    assert errors == [], "Validation should pass with no errors"
    assert result == {"name": "John Doe", "age": 30}, "Result should match the expected schema"

    invalid_content = """
    name: John Doe
    age: "thirty"
    """
    result, errors = validate_yaml(invalid_content, ExampleSchema)
    assert errors != [], "Validation should fail due to incorrect age type"
    assert isinstance(errors[0], ParseError), "Errors should contain ParseError instances"


# LLM-generated content at query #25
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    """
    class Person(Schema):
        name = Field(str)
        age = Field(int)
    result = validate_yaml(content, Person)
    assert isinstance(result, Person)
    assert result.name == "John Doe"
    assert result.age == 30


# LLM-generated content at query #26
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: John Doe
    age: 30
    """

    result, errors = validate_yaml(yaml_content, TestSchema)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30

    invalid_yaml_content = """
    name: John Doe
    age: "thirty"
    """

    result, errors = validate_yaml(invalid_yaml_content, TestSchema)
    assert len(errors) > 0
    assert errors[0].code == "type_error"

    empty_yaml_content = ""
    try:
        result, errors = validate_yaml(empty_yaml_content, TestSchema)
    except ParseError as e:
        assert e.code == "no_content"

    malformed_yaml_content = """
    name: John Doe
    age: thirty
    """

    try:
        result, errors = validate_yaml(malformed_yaml_content, TestSchema)
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #27
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class Person(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    yaml_content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(yaml_content, Person)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = """
    name: John Doe
    age: thirty
    """
    try:
        validate_yaml(invalid_yaml, Person)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test validation error
    invalid_content = """
    name: John Doe
    age: 200
    """
    value, errors = validate_yaml(invalid_content, Person)
    assert len(errors) == 1
    assert errors[0].text == "Must be less than or equal to 150."

    # Test empty YAML
    empty_content = ""
    try:
        validate_yaml(empty_content, Person)
    except ParseError as exc:
        assert exc.code == "no_content"

    # Test bytes input
    bytes_content = b"""
    name: Jane Doe
    age: 25
    """
    value, errors = validate_yaml(bytes_content, Person)
    assert errors == []
    assert value == {"name": "Jane Doe", "age": 25}


# LLM-generated content at query #28
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    content = """
    name: John Doe
    age: 30
    """
    token = tokenize_yaml(content)
    assert isinstance(token, DictToken)
    assert isinstance(token.value["name"], ScalarToken)
    assert token.value["name"].value == "John Doe"
    assert token.value["age"].value == 30



# LLM-generated content at query #29
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    yaml_content = """
    name: John Doe
    age: 30
    """
    value, error_messages = validate_yaml(yaml_content, ExampleSchema)
    assert value == {"name": "John Doe", "age": 30}
    assert error_messages == []

    invalid_yaml_content = """
    name: John Doe
    age: '30'
    """
    value, error_messages = validate_yaml(invalid_yaml_content, ExampleSchema)
    assert value is None
    assert len(error_messages) == 1
    assert error_messages[0].text == "Must be of type 'int'."

    invalid_yaml_syntax = """
    name: John Doe
    age: 30
    extra_field: value
    """
    value, error_messages = validate_yaml(invalid_yaml_syntax, ExampleSchema)
    assert value is None
    assert len(error_messages) == 1
    assert error_messages[0].text == "Unknown field 'extra_field'."


# LLM-generated content at query #30
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class Person(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, Person)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John\nage: thirty"
    try:
        validate_yaml(invalid_yaml, Person)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test validation error
    invalid_data = "name: John\nage: -5"
    value, errors = validate_yaml(invalid_data, Person)
    assert len(errors) == 1
    assert errors[0].code == "minimum"

    # Test empty content
    empty_content = ""
    try:
        validate_yaml(empty_content, Person)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #31
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    content = "name: John\nage: 30"
    value, errors = validate_yaml(content, ExampleSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    content = "name: John\nage: thirty"
    value, errors = validate_yaml(content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "type_error"
    assert errors[0].text == "Must be of type 'int'."
    assert errors[0].position.line_no == 2
    assert errors[0].position.column_no == 5

    content = ""
    try:
        validate_yaml(content, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"
        assert exc.text == "No content."
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1

    content = "name: John\nage: 30\ninvalid: field"
    value, errors = validate_yaml(content, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].code == "invalid_key"
    assert errors[0].text == "Invalid key."
    assert errors[0].position.line_no == 3
    assert errors[0].position.column_no == 1


# LLM-generated content at query #32
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test case 1: Valid YAML content with a simple schema
    content = 'name: "John Doe"'
    schema = Schema.from_dict({"name": Field(str)})
    validated_data, errors = validate_yaml(content, schema)
    assert validated_data == {"name": "John Doe"}
    assert errors == []

    # Test case 2: Invalid YAML content with a schema
    content = 'name: 123'
    schema = Schema.from_dict({"name": Field(str)})
    validated_data, errors = validate_yaml(content, schema)
    assert validated_data is None
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'string'."

    # Test case 3: Invalid YAML format
    content = 'name: John Doe:'
    schema = Schema.from_dict({"name": Field(str)})
    try:
        validate_yaml(content, schema)
    except ParseError as e:
        assert e.text == "could not find expected ':'."
        assert e.code == "parse_error"

    # Test case 4: Empty YAML content
    content = ''
    schema = Schema.from_dict({"name": Field(str)})
    try:
        validate_yaml(content, schema)
    except ParseError as e:
        assert e.text == "No content."
        assert e.code == "no_content"

    # Test case 5: YAML content with nested schema validation
    content = '''
    person:
      name: "John Doe"
      age: 30
    '''
    schema = Schema.from_dict({
        "person": Schema.from_dict({
            "name": Field(str),
            "age": Field(int)
        })
    })
    validated_data, errors = validate_yaml(content, schema)
    assert validated_data == {"person": {"name": "John Doe", "age": 30}}
    assert errors == []

    # Test case 6: YAML content with nested schema validation failure
    content = '''
    person:
      name: "John Doe"
      age: "thirty"
    '''
    schema = Schema.from_dict({
        "person": Schema.from_dict({
            "name": Field(str),
            "age": Field(int)
        })
    })
    validated_data, errors = validate_yaml(content, schema)
    assert validated_data is None
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'integer'."


# LLM-generated content at query #33
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    content = """
    name: John Doe
    age: 30
    """
    value, errors = validate_yaml(content, TestSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    content = """
    name: John Doe
    age: thirty
    """
    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."
    assert errors[0].code == "type_error.integer"

    # Test invalid YAML (validation error)
    content = """
    name: John Doe
    age: -5
    """
    value, errors = validate_yaml(content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be greater than or equal to 0."
    assert errors[0].code == "min_value"

    # Test empty YAML
    content = ""
    try:
        validate_yaml(content, TestSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    print("All tests passed!")

test_validate_yaml()


# LLM-generated content at query #34
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML content
    valid_content = "name: John\nage: 30"
    value, errors = validate_yaml(valid_content, TestSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML content (missing required field)
    invalid_content = "name: John"
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "The field 'age' is required."

    # Test invalid YAML content (wrong type)
    invalid_content = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(invalid_content, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'int'."

    # Test empty YAML content
    empty_content = ""
    try:
        validate_yaml(empty_content, TestSchema)
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"

    # Test invalid YAML syntax
    invalid_syntax = "name: John\nage: 30:"
    try:
        validate_yaml(invalid_syntax, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"


# LLM-generated content at query #35
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty content
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\nkey2: value2\nkey3: value3\nkey4: value4\nkey5: value5\nkey6: value6\nkey7: value7\nkey8: value8\nkey9: value9\nkey10: value10\nkey11: value11\nkey12: value12\nkey13: value13\nkey14: value14\nkey15: value15\nkey16: value16\nkey17: value17\nkey18: value18\nkey19: value19\nkey20: value20\nkey21: value21\nkey22: value22\nkey23: value23\nkey24: value24\nkey25: value25\nkey26: value26\nkey27: value27\nkey28: value28\nkey29: value29\nkey30: value30\nkey31: value31\nkey32: value32\nkey33: value33\nkey34: value34\nkey35: value35\nkey36: value36\nkey37: value37\nkey38: value38\nkey39: value39\nkey40: value40\nkey41: value41\nkey42: value42\nkey43: value43\nkey44: value44\nkey45: value45\nkey46: value46\nkey47: value47\nkey48: value48\nkey49: value49\nkey50: value50\nkey51: value51\nkey52: value52\nkey53: value53\nkey54: value54\nkey55: value55\nkey56: value56\nkey57: value57\nkey58: value58\nkey59: value59\nkey60: value60\nkey61: value61\nkey62: value62\nkey63: value63\nkey64: value64\nkey65: value65\nkey66: value66\nkey67: value67\nkey68: value68\nkey69: value69\nkey70: value70\nkey71: value71\nkey72: value72\nkey73: value73\nkey74: value74\nkey75: value75\nkey76: value76\nkey77: value77\nkey78: value78\nkey79: value79\nkey80: value80\nkey81: value81\nkey82: value82\nkey83: value83\nkey84: value84\nkey85: value85\nkey86: value86\nkey87: value87\nkey88: value88\nkey89: value89\nkey90: value90\nkey91: value91\nkey92: value92\nkey93: value93\nkey94: value94\nkey95: value95\nkey96: value96\nkey97: value97\nkey98: value98\nkey99: value99\nkey100: value100\nkey101: value101\nkey102: value102\nkey103: value103\nkey104: value104\nkey105: value105\nkey106: value106\nkey107: value107\nkey108: value108\nkey109: value109\nkey110: value110\nkey111: value111\nkey112: value112\nkey113: value113\nkey114: value114\nkey115: value115\nkey116: value116\nkey117: value117\nkey118: value118\nkey119: value119\nkey120: value120\nkey121: value121\nkey122: value122\nkey123: value123\nkey124: value124\nkey125: value125\nkey126: value126\nkey127: value127\nkey128: value128\nkey129: value129\nkey130: value130\nkey131: value131\nkey132: value132\nkey133: value133\nkey134: value134\nkey135: value135\nkey136: value136\nkey137: value137\nkey138: value138\nkey139: value139\nkey140: value140\nkey141: value141\nkey142: value142\nkey143: value143\nkey144: value144\nkey145: value145\nkey146: value146\nkey147: value147\nkey148: value148\nkey149: value149\nkey150: value150\nkey151: value151\nkey152: value152\nkey153: value153\nkey154: value154\nkey155: value155\nkey156: value156\nkey157: value157\nkey158: value158\nkey159: value159\nkey160: value160\nkey161: value161\nkey162: value162\nkey163: value163\nkey164: value164\nkey165: value165\nkey166: value166\nkey167: value167\nkey168: value168\nkey169: value169\nkey170: value170\nkey171: value171\nkey172: value172\nkey173: value173\nkey174: value174\nkey175: value175\nkey176: value176\nkey177: value177\nkey178: value178\nkey179: value179\nkey180: value180\nkey181: value181\nkey182: value182\nkey183: value183\nkey184: value184\nkey185: value185\nkey186: value186\nkey187: value187\nkey188: value188\nkey189: value189\nkey190: value190\nkey191: value191\nkey192: value192\nkey193: value193\nkey194: value194\nkey195: value195\nkey196: value196\nkey197: value197\nkey198: value198\nkey199: value199\nkey200: value200\nkey201: value201\nkey202: value202\nkey203: value203\nkey204: value204\nkey205: value205\nkey206: value206\nkey207: value207\nkey208: value208\nkey209: value209\nkey210: value210\nkey211: value211\nkey212: value212\nkey213: value213\nkey214: value214\nkey215: value215\nkey216: value216\nkey217: value217\nkey218: value218\nkey219: value219\nkey220: value220\nkey221: value221\nkey222: value222\nkey223: value223\nkey224: value224\nkey225: value225\nkey226: value226\nkey227: value227\nkey228: value228\nkey229: value229\nkey230: value230\nkey231: value231\nkey232: value232\nkey233: value233\nkey234: value234\nkey235: value235\nkey236: value236\nkey237: value237\nkey238: value238\nkey239: value239\nkey240: value240\nkey241: value241\nkey242: value242\nkey243: value243\nkey244: value244\nkey245: value245\nkey246: value246\nkey247: value247\nkey248: value248\nkey249: value249\nkey250: value250\nkey251: value251\nkey252: value252\nkey253: value253\nkey254: value254\nkey255: value255\nkey256: value256\nkey257: value257\nkey258: value258\nkey259: value259\nkey260: value260\nkey261: value261\nkey262: value262\nkey263: value263\nkey264: value264\nkey265: value265\nkey266: value266\nkey267: value267\nkey268: value268\nkey269: value269\nkey270: value270\nkey271: value271\nkey272: value272\nkey273: value273\nkey274: value274\nkey275: value275\nkey276: value276\nkey277: value277\nkey278: value278\nkey279: value279\nkey280: value280\nkey281: value281\nkey282: value282\nkey283: value283\nkey284: value284\nkey285: value285\nkey286: value286\nkey287: value287\nkey288: value288\nkey289: value289\nkey290: value290\nkey291: value291\nkey292: value292\nkey293: value293\nkey294: value294\nkey295: value295\nkey296: value296\nkey297: value297\nkey298: value298\nkey299: value299\nkey300: value300\nkey301: value301\nkey302: value302\nkey303: value303\nkey304: value304\nkey305: value305\nkey306: value306\nkey307: value307\nkey308: value308\nkey309: value309\nkey310: value310\nkey311: value311\nkey312: value312\nkey313: value313\nkey314: value314\nkey315: value315\nkey316: value316\nkey317: value317\nkey318: value318\nkey319: value319\nkey320: value320\nkey321: value321\nkey322: value322\nkey323: value323\nkey324: value324\nkey325: value325\nkey326: value326\nkey327: value327


# LLM-generated content at query #36
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String, Schema

    class TestSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (parse error)
    invalid_yaml = "name: John Doe\nage: "
    try:
        validate_yaml(invalid_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test invalid YAML (validation error)
    invalid_yaml = "name: John Doe\nage: 200"
    value, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].code == "maximum"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, TestSchema)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #37
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class TestSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    result, errors = validate_yaml(valid_yaml, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test invalid YAML (missing required field)
    invalid_yaml = "name: John"
    result, errors = validate_yaml(invalid_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "The field 'age' is required."

    # Test invalid YAML (invalid type)
    invalid_type_yaml = "name: John\nage: thirty"
    result, errors = validate_yaml(invalid_type_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."

    # Test empty YAML
    empty_yaml = ""
    result, errors = validate_yaml(empty_yaml, TestSchema)
    assert len(errors) == 1
    assert errors[0].text == "No content."

    # Test YAML with extra fields
    extra_fields_yaml = "name: John\nage: 30\ncity: New York"
    result, errors = validate_yaml(extra_fields_yaml, TestSchema)
    assert errors == []
    assert result == {"name": "John", "age": 30}

    # Test YAML with nested structure
    class NestedSchema(Schema):
        person = TestSchema

    nested_yaml = "person:\n  name: John\n  age: 30"
    result, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert result == {"person": {"name": "John", "age": 30}}


# LLM-generated content at query #38
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    content = """
    name: John Doe
    age: 30
    is_student: false
    """
    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)
        is_student = Field(bool)
    
    validator = PersonSchema()
    result, error_messages = validate_yaml(content, validator)
    assert error_messages == []
    assert result == {"name": "John Doe", "age": 30, "is_student": False}


# LLM-generated content at query #39
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)
    
    valid_yaml = "name: John\nage: 30"
    expected_output = {"name": "John", "age": 30}
    result, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert result == expected_output
    assert errors == []

    invalid_yaml = "name: John\nage: thirty"
    result, errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert result == {"name": "John", "age": "thirty"}
    assert len(errors) == 1
    assert isinstance(errors[0], ParseError)
    assert errors[0].code == "type_error.integer"

    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as e:
        assert e.code == "no_content"
        assert e.text == "No content."

    invalid_yaml_structure = "- name: John\n  age: 30"
    result, errors = validate_yaml(invalid_yaml_structure, ExampleSchema)
    assert len(errors) == 1
    assert isinstance(errors[0], ParseError)
    assert errors[0].code == "type_error.dict"

    invalid_yaml_syntax = "name: John\nage: : 30"
    try:
        validate_yaml(invalid_yaml_syntax, ExampleSchema)
    except ParseError as e:
        assert e.code == "parse_error"


# LLM-generated content at query #40
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class ExampleSchema(Schema):
        name = Field(str)
        age = Field(int)

    valid_yaml = "name: John\nage: 30"
    invalid_yaml = "name: John\nage: 'thirty'"

    valid_result, valid_errors = validate_yaml(valid_yaml, ExampleSchema)
    assert valid_errors == []

    invalid_result, invalid_errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert len(invalid_errors) == 1
    assert invalid_errors[0].code == "type_error"


# LLM-generated content at query #41
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    class UserSchema(Schema):
        name = Field(str)
        age = Field(int)

    # Test valid YAML
    valid_yaml = "name: John\nage: 30"
    value, errors = validate_yaml(valid_yaml, UserSchema)
    assert errors == []
    assert value == {"name": "John", "age": 30}

    # Test invalid YAML (missing required field)
    invalid_yaml = "name: John"
    value, errors = validate_yaml(invalid_yaml, UserSchema)
    assert len(errors) == 1
    assert errors[0].text == "The field 'age' is required."

    # Test invalid YAML (wrong data type)
    invalid_yaml = "name: John\nage: 'thirty'"
    value, errors = validate_yaml(invalid_yaml, UserSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be a number."

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, UserSchema)
    except ParseError as e:
        assert e.text == "No content."

    # Test YAML with syntax error
    invalid_syntax_yaml = "name: John\nage: thirty:"
    try:
        validate_yaml(invalid_syntax_yaml, UserSchema)
    except ParseError as e:
        assert e.text == "found unexpected ':' while scanning a simple key."

    # Test YAML with nested structure
    nested_yaml = "user:\n  name: John\n  age: 30"
    class NestedSchema(Schema):
        user = Field(UserSchema)
    value, errors = validate_yaml(nested_yaml, NestedSchema)
    assert errors == []
    assert value == {"user": {"name": "John", "age": 30}}

    # Test YAML with list
    list_yaml = "users:\n  - name: John\n    age: 30\n  - name: Jane\n    age: 25"
    class ListSchema(Schema):
        users = Field(typing.List[UserSchema])
    value, errors = validate_yaml(list_yaml, ListSchema)
    assert errors == []
    assert value == {"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}


# LLM-generated content at query #42
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test case 1: Valid YAML with correct schema
    content = """
    name: John Doe
    age: 30
    """
    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)

    result, errors = validate_yaml(content, PersonSchema)
    assert errors == []
    assert result == {"name": "John Doe", "age": 30}

    # Test case 2: Invalid YAML with incorrect schema
    content = """
    name: John Doe
    age: "30"
    """
    result, errors = validate_yaml(content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'int'."

    # Test case 3: Invalid YAML with incorrect structure
    content = """
    name: John Doe
    age:
      - 30
      - 40
    """
    result, errors = validate_yaml(content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be of type 'int'."

    # Test case 4: Empty YAML
    content = ""
    result, errors = validate_yaml(content, PersonSchema)
    assert len(errors) == 1
    assert errors[0].text == "No content."

    # Test case 5: Valid YAML with nested structure
    content = """
    name: John Doe
    age: 30
    address:
      city: New York
      zip: 10001
    """
    class AddressSchema(Schema):
        city = Field(str)
        zip = Field(int)

    class PersonSchema(Schema):
        name = Field(str)
        age = Field(int)
        address = Field(AddressSchema)

    result, errors = validate_yaml(content, PersonSchema)
    assert errors == []
    assert result == {"name": "John Doe", "age": 30, "address": {"city": "New York", "zip": 10001}}


# LLM-generated content at query #43
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    from typesystem import Integer, String

    class ExampleSchema(Schema):
        name = String(max_length=100)
        age = Integer(minimum=0, maximum=150)

    # Test valid YAML
    valid_yaml = "name: John Doe\nage: 30"
    value, errors = validate_yaml(valid_yaml, ExampleSchema)
    assert errors == []
    assert value == {"name": "John Doe", "age": 30}

    # Test invalid YAML (age out of range)
    invalid_yaml = "name: John Doe\nage: 200"
    value, errors = validate_yaml(invalid_yaml, ExampleSchema)
    assert len(errors) == 1
    assert errors[0].text == "Must be less than or equal to 150."

    # Test invalid YAML (syntax error)
    invalid_syntax_yaml = "name: John Doe\nage:"
    try:
        validate_yaml(invalid_syntax_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "parse_error"

    # Test empty YAML
    empty_yaml = ""
    try:
        validate_yaml(empty_yaml, ExampleSchema)
    except ParseError as exc:
        assert exc.code == "no_content"


# LLM-generated content at query #44
#--------------------------

# Unit test for function tokenize_yaml
def test_tokenize_yaml():
    # Test empty string
    try:
        tokenize_yaml("")
    except ParseError as exc:
        assert exc.text == "No content."
        assert exc.code == "no_content"
        assert exc.position.line_no == 1
        assert exc.position.column_no == 1
        assert exc.position.char_index == 0

    # Test invalid YAML
    try:
        tokenize_yaml("key: value\nkey2: value: value2")
    except ParseError as exc:
        assert exc.text == "could not find expected ':'."
        assert exc.code == "parse_error"

    # Test valid YAML
    token = tokenize_yaml("key: value")
    assert isinstance(token, DictToken)
    assert token.value == {"key": "value"}
    assert token.start == 0
    assert token.end == 9
    assert token.content == "key: value"

    # Test scalar tokens
    token = tokenize_yaml("123")
    assert isinstance(token, ScalarToken)
    assert token.value == 123
    assert token.start == 0
    assert token.end == 2
    assert token.content == "123"

    # Test list tokens
    token = tokenize_yaml("- item1\n- item2")
    assert isinstance(token, ListToken)
    assert token.value == ["item1", "item2"]
    assert token.start == 0
    assert token.end == 13
    assert token.content == "- item1\n- item2"


# LLM-generated content at query #45
#--------------------------

# Unit test for function validate_yaml
def test_validate_yaml():
    # Test with valid YAML content
    content = """
    name: John Doe
    age: 30
    """
    validator = Schema.from_dict({
        "name": Field(str),
        "age": Field(int)
    })
    result, errors = validate_yaml(content, validator)
    assert errors == []
    assert result["name"] == "John Doe"
    assert result["age"] == 30

    # Test with invalid YAML content
    content = """
    name: John Doe
    age: thirty
    """
    result, errors = validate_yaml(content, validator)
    assert len(errors) > 0
    assert errors[0].code == "type_error"

    # Test with empty YAML content
    content = ""
    result, errors = validate_yaml(content, validator)
    assert len(errors) > 0
    assert errors[0].code == "no_content"

    # Test with invalid YAML syntax
    content = """
    name: John Doe
    age: 30
    """
    validator = Schema.from_dict({
        "name": Field(str),
        "age": Field(int)
    })
    # Introduce a syntax error in the YAML content
    content = """
    name: John Doe
    age: 30
    invalid:
    """
    result, errors = validate_yaml(content, validator)
    assert len(errors) > 0
    assert errors[0].code == "parse_error"


