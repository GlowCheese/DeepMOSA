####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions_valid_value():
    class MockField:
        def validate(self, value):
            return value

    token = Token(value={"key": "value"}, start_index=0, end_index=10, content='{"key": "value"}')
    validator = MockField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"key": "value"}

def test_validate_with_positions_required_error():
    class MockSchema:
        def validate(self, value):
            raise ValidationError(messages=[Message(text="This field is required.", code="required", index=["key"])])

    token = Token(value={}, start_index=0, end_index=1, content="{}")
    validator = MockSchema()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "The field 'key' is required."
        assert message.code == "required"
        assert message.index == ["key"]

def test_validate_with_positions_custom_error():
    class MockField:
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Invalid value.", code="custom", index=["key"])])

    token = Token(value={"key": "invalid"}, start_index=0, end_index=15, content='{"key": "invalid"}')
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "Invalid value."
        assert message.code == "custom"
        assert message.index == ["key"]


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions_valid():
    token = Token(value={"username": "test"}, start_index=0, end_index=20, content='{"username": "test"}')
    schema = Schema(fields={"username": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"username": "test"}

def test_validate_with_positions_missing_required_field():
    token = Token(value={}, start_index=0, end_index=2, content='{}')
    schema = Schema(fields={"username": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'username' is required."
        assert message.code == "required"
        assert message.index == ["username"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 1

def test_validate_with_positions_invalid_type():
    token = Token(value="invalid", start_index=0, end_index=7, content='"invalid"')
    schema = Schema(fields={"username": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 7


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError

    class TestToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self

    token = TestToken(value={}, start_index=0, end_index=0)
    schema = Schema(fields={"required_field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert any(msg.code == "required" for msg in error.messages())


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    class MockField:
        def validate(self, value):
            raise ValidationError(text="Validation failed", code="custom")

    token = Token(value="test", start_index=0, end_index=3, content="test")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert error.messages()[0].code == "custom"
        assert error.messages()[0].text == "Validation failed"


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions_valid_value():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == {"name": "John"}

def test_validate_with_positions_null_value():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    field = Field(allow_null=True)
    result = validate_with_positions(token=token, validator=field)
    assert result is None

def test_validate_with_positions_required_field_missing():
    token = Token(value={}, start_index=0, end_index=2, content="{}")
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'name' is required."
        assert message.code == "required"
        assert message.index == ["name"]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 3

def test_validate_with_positions_invalid_key():
    token = Token(value={1: "John"}, start_index=0, end_index=10, content='{1: "John"}')
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "All object keys must be strings."
        assert message.code == "invalid_key"
        assert message.index == [1]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 11

def test_validate_with_positions_multiple_errors():
    token = Token(value={"age": "twenty"}, start_index=0, end_index=15, content='{"age": "twenty"}')
    schema = Schema(fields={"age": Field(), "name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 2
        messages = error.messages()
        assert messages[0].text == "The field 'name' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["name"]
        assert messages[0].start_position.line_no == 1
        assert messages[0].start_position.column_no == 1
        assert messages[0].end_position.line_no == 1
        assert messages[0].end_position.column_no == 16
        assert messages[1].text == "Must be an object."
        assert messages[1].code == "type"
        assert messages[1].index == []
        assert messages[1].start_position.line_no == 1
        assert messages[1].start_position.column_no == 1
        assert messages[1].end_position.line_no == 1
        assert messages[1].end_position.column_no == 16


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions_with_required_field_error():
    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}

        def _get_child_token(self, key):
            if key == "age":
                return MockToken(None, 10, 20, "some content")
            return MockToken(None, 0, 9, "some content")

        def _get_key_token(self, key):
            return MockToken(None, 0, 9, "some content")

    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 50, "some content")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"
        assert message.index == ["age"]
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 20

def test_validate_with_positions_with_validation_error():
    class MockToken(Token):
        def _get_value(self):
            return {"age": "invalid"}

        def _get_child_token(self, key):
            if key == "age":
                return MockToken(None, 10, 20, "some content")
            return MockToken(None, 0, 9, "some content")

        def _get_key_token(self, key):
            return MockToken(None, 0, 9, "some content")

    fields = {"age": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 50, "some content")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "Must be a valid value."
        assert message.code == "type"
        assert message.index == ["age"]
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 20

def test_validate_with_positions_with_valid_value():
    class MockToken(Token):
        def _get_value(self):
            return {"age": 25}

        def _get_child_token(self, key):
            return MockToken(None, 0, 9, "some content")

        def _get_key_token(self, key):
            return MockToken(None, 0, 9, "some content")

    fields = {"age": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 50, "some content")
    validated_value = validate_with_positions(token=token, validator=schema)
    assert validated_value == {"age": 25}


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions_valid_value():
    class StringField(Field):
        def validate(self, value: typing.Any) -> typing.Any:
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    token = Token(value="valid", start_index=0, end_index=4, content="valid")
    validator = StringField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid"


def test_validate_with_positions_invalid_value():
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value: typing.Any) -> typing.Any:
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    token = Token(value=123, start_index=0, end_index=2, content="123")
    validator = StringField()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        message = error.messages()[0]
        assert message.text == "Must be a string."
        assert message.code == "type"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 2


def test_validate_with_positions_required_field():
    class RequiredField(Field):
        errors = {"required": "This field is required."}

    token = Token(value={}, start_index=0, end_index=1, content="{}")
    validator = Schema(fields={"field": RequiredField()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        message = error.messages()[0]
        assert message.text == "The field 'field' is required."
        assert message.code == "required"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 1


def test_validate_with_positions_nested_required_field():
    class RequiredField(Field):
        errors = {"required": "This field is required."}

    token = Token(value={"nested": {}}, start_index=0, end_index=13, content='{"nested": {}}')
    validator = Schema(fields={"nested": Schema(fields={"field": RequiredField()})})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        message = error.messages()[0]
        assert message.text == "The field 'field' is required."
        assert message.code == "required"
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 12


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions_required_field():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content="")
    validator = Schema(fields={"age": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"
        assert message.index == ["age"]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 1
        assert message.end_position.char_index == 0

def test_validate_with_positions_invalid_type():
    token = Token(value="not an object", start_index=0, end_index=12, content="")
    validator = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 13
        assert message.end_position.char_index == 12

def test_validate_with_positions_nested_validation_error():
    token = Token(value={"user": {"name": 123}}, start_index=0, end_index=18, content="")
    validator = Schema(fields={"user": Schema(fields={"name": Field()})})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "Must be a string."
        assert message.code == "type"
        assert message.index == ["user", "name"]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 19
        assert message.end_position.char_index == 18


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions_required_field():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"
        assert message.index == ["age"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_invalid_type():
    token = Token(value={"name": 123}, start_index=0, end_index=10, content='{"name": 123}')
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_invalid_key():
    token = Token(value={123: "John"}, start_index=0, end_index=10, content='{123: "John"}')
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "All object keys must be strings."
        assert message.code == "invalid_key"
        assert message.index == [123]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_null_value():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages) == 1
        message = error.messages[0]
        assert message.text == "May not be null."
        assert message.code == "null"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 4


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    validator = Schema(fields={"field": Field()})
    validator.validate(token.value)


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    class MockToken:
        def __init__(self, value):
            self._value = value
        
        def value(self):
            return self._value
        
        def lookup(self, index):
            return self
    
    class MockValidator:
        def validate(self, value):
            raise ValidationError(text="Test error", code="test_code")
    
    token = MockToken("test_value")
    validator = MockValidator()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError:
        pass
    else:
        assert False, "Expected ValidationError to be raised"


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions_required_field():
    class MockToken(Token):
        def _get_value(self):
            return {"existing_field": "value"}

        def lookup(self, index):
            return MockToken("", 0, 0)

        def _get_position(self, index):
            return Position(1, 1, index)

    token = MockToken({"existing_field": "value"}, 0, 0)
    schema = Schema(fields={"required_field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'required_field' is required."
        assert message.code == "required"
        assert message.index == ["required_field"]
        assert message.start_position == Position(1, 1, 0)
        assert message.end_position == Position(1, 1, 0)


def test_validate_with_positions_invalid_field():
    class MockToken(Token):
        def _get_value(self):
            return {"invalid_field": 123}

        def lookup(self, index):
            return MockToken("", 0, 0)

        def _get_position(self, index):
            return Position(1, 1, index)

    token = MockToken({"invalid_field": 123}, 0, 0)
    schema = Schema(fields={"invalid_field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == ["invalid_field"]
        assert message.start_position == Position(1, 1, 0)
        assert message.end_position == Position(1, 1, 0)


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions_passes_predicate():
    token = Token(value={}, start_index=0, end_index=0, content="{}")
    validator = Field()
    try:
        validator.validate(token.value)
    except ValidationError as error:
        assert isinstance(error, ValidationError)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_required_field():
    token = Token(value={"field": "value"}, start_index=0, end_index=10, content='{"field": "value"}')
    schema = Schema(fields={"missing_field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.text == "The field 'missing_field' is required."
        assert message.code == "required"
        assert message.index == ["missing_field"]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 1
        assert message.end_position.char_index == 0

def test_validate_with_positions_invalid_type():
    token = Token(value="not_a_dict", start_index=0, end_index=10, content='"not_a_dict"')
    schema = Schema(fields={"field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 1
        assert message.end_position.char_index == 0

def test_validate_with_positions_valid_value():
    token = Token(value={"field": "value"}, start_index=0, end_index=10, content='{"field": "value"}')
    schema = Schema(fields={"field": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "value"}

def test_validate_with_positions_null_value():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    schema = Schema(fields={"field": Field(allow_null=True)})
    result = validate_with_positions(token=token, validator=schema)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    token = Token(value=None, start_index=0, end_index=0)
    validator = Field(allow_null=False)
    validator.validate(token.value)


# LLM-generated content at query #16
#--------------------------

def test_validate_with_positions_required_field():
    token = Token(value={"name": "test"}, start_index=0, end_index=10, content='{"name": "test"}')
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 10

def test_validate_with_positions_invalid_type():
    token = Token(value="invalid", start_index=0, end_index=7, content='"invalid"')
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == []
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 7

def test_validate_with_positions_nested_validation():
    token = Token(
        value={"user": {"name": 123}},
        start_index=0,
        end_index=20,
        content='{"user": {"name": 123}}'
    )
    schema = Schema(fields={"user": Schema(fields={"name": Field()})})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == ["user", "name"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 20

def test_validate_with_positions_successful_validation():
    token = Token(
        value={"name": "test", "age": 25},
        start_index=0,
        end_index=20,
        content='{"name": "test", "age": 25}'
    )
    schema = Schema(fields={"name": Field(), "age": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test", "age": 25}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    class MockToken(Token):
        def _get_value(self):
            return {}
        def lookup(self, index):
            return MockToken(None, 0, 5, "field")

    field = Field(read_only=False)
    schema = Schema(fields={"field": field})
    token = MockToken(None, 0, 10, "field")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'field' is required."
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 5

def test_validate_with_positions_type_error():
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def lookup(self, index):
            return MockToken(None, 0, 5, "field")

    field = Field(read_only=False)
    schema = Schema(fields={"field": field})
    token = MockToken(None, 0, 10, "field")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_null_error():
    class MockToken(Token):
        def _get_value(self):
            return None
        def lookup(self, index):
            return MockToken(None, 0, 5, "field")

    field = Field(read_only=False)
    schema = Schema(fields={"field": field})
    token = MockToken(None, 0, 10, "field")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "May not be null."
        assert message.code == "null"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    class MockToken(Token):
        def _get_value(self):
            return {}

        def lookup(self, index):
            return MockToken(value=None, start_index=0, end_index=5)

    token = MockToken(value=None, start_index=0, end_index=5)
    schema = Schema(fields={"field_name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        message = error.messages[0]
        assert message.text == "The field 'field_name' is required."
        assert message.code == "required"
        assert message.index == ["field_name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 5

def test_validate_with_positions_generic_error():
    class MockToken(Token):
        def _get_value(self):
            return {"field_name": "invalid_value"}

        def lookup(self, index):
            return MockToken(value=None, start_index=10, end_index=20)

    token = MockToken(value=None, start_index=0, end_index=20)
    schema = Schema(fields={"field_name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        message = error.messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 20


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions_valid_input():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_missing_required_field():
    token = Token(value={}, start_index=0, end_index=2, content="{}")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        message = error.messages[0]
        assert message.text == "The field 'name' is required."
        assert message.code == "required"
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 1

def test_validate_with_positions_invalid_type():
    token = Token(value=123, start_index=0, end_index=3, content="123")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        message = error.messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 2

def test_validate_with_positions_nested_field_error():
    token = Token(value={"user": {"name": 123}}, start_index=0, end_index=20, content='{"user": {"name": 123}}')
    fields = {"user": Schema(fields={"name": Field()})}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        message = error.messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == ["user", "name"]
        assert message.start_position.char_index == 9
        assert message.end_position.char_index == 19


# LLM-generated content at query #4
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}

        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

    fields = {"name": String(), "age": String(required=True)}
    schema = Schema(fields=fields)
    token = MockToken(value=None, start_index=0, end_index=10)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
        assert messages[0].text == "The field 'age' is required."


def test_validate_with_positions_invalid_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Integer
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"age": "not_an_integer"}

        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

    fields = {"age": Integer()}
    schema = Schema(fields=fields)
    token = MockToken(value=None, start_index=0, end_index=10)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == ["age"]


def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}

        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=10)

    fields = {"name": String()}
    schema = Schema(fields=fields)
    token = MockToken(value=None, start_index=0, end_index=10)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    field = Field()
    token = Token(value=None, start_index=0, end_index=0)
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError:
        pass
    else:
        assert False, "Expected ValidationError to be raised"


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions_with_required_error():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content="{'name': 'John'}")
    schema = Schema(fields={"age": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"
        assert message.index == ["age"]
        assert message.start_position.line == 1
        assert message.start_position.column == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line == 1
        assert message.end_position.column == 16
        assert message.end_position.char_index == 15

def test_validate_with_positions_with_invalid_key_error():
    token = Token(value={1: "John"}, start_index=0, end_index=9, content="{1: 'John'}")
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "All object keys must be strings."
        assert message.code == "invalid_key"
        assert message.index == [1]
        assert message.start_position.line == 1
        assert message.start_position.column == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line == 1
        assert message.end_position.column == 10
        assert message.end_position.char_index == 9

def test_validate_with_positions_with_type_error():
    token = Token(value="John", start_index=0, end_index=4, content="'John'")
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.line == 1
        assert message.start_position.column == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line == 1
        assert message.end_position.column == 6
        assert message.end_position.char_index == 5

def test_validate_with_positions_with_null_error():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "May not be null."
        assert message.code == "null"
        assert message.index == []
        assert message.start_position.line == 1
        assert message.start_position.column == 1
        assert message.start_position.char_index == 0
        assert message.end_position.line == 1
        assert message.end_position.column == 5
        assert message.end_position.char_index == 4


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions_required_field():
    class MockToken:
        def __init__(self, value, start_index, end_index):
            self._value = value
            self._start_index = start_index
            self._end_index = end_index

        def lookup(self, index):
            return self

        @property
        def start(self):
            return {"char_index": self._start_index}

        @property
        def end(self):
            return {"char_index": self._end_index}

        @property
        def value(self):
            return self._value

    class MockValidator:
        def validate(self, value):
            raise ValidationError(messages=[Message(text="required", code="required", index=["field"])])

    token = MockToken(value={}, start_index=0, end_index=10)
    validator = MockValidator()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'field' is required."
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.start_position == {"char_index": 0}
        assert message.end_position == {"char_index": 10}

def test_validate_with_positions_non_required_field():
    class MockToken:
        def __init__(self, value, start_index, end_index):
            self._value = value
            self._start_index = start_index
            self._end_index = end_index

        def lookup(self, index):
            return self

        @property
        def start(self):
            return {"char_index": self._start_index}

        @property
        def end(self):
            return {"char_index": self._end_index}

        @property
        def value(self):
            return self._value

    class MockValidator:
        def validate(self, value):
            raise ValidationError(messages=[Message(text="invalid", code="invalid", index=["field"])])

    token = MockToken(value={}, start_index=5, end_index=15)
    validator = MockValidator()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "invalid"
        assert message.code == "invalid"
        assert message.index == ["field"]
        assert message.start_position == {"char_index": 5}
        assert message.end_position == {"char_index": 15}


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions_valid_value():
    class MockToken:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    class MockValidator:
        def validate(self, value):
            return value

    token = MockToken("valid")
    validator = MockValidator()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid"

def test_validate_with_positions_validation_error():
    class MockToken:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def lookup(self, index):
            return self

        def start(self):
            return 0

        def end(self):
            return 1

    class MockValidator:
        def validate(self, value):
            raise ValidationError(text="Invalid value", code="invalid")

    token = MockToken("invalid")
    validator = MockValidator()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].text == "Invalid value"
        assert error.messages()[0].code == "invalid"
        assert error.messages()[0].start_position == 0
        assert error.messages()[0].end_position == 1

def test_validate_with_positions_required_field_error():
    class MockToken:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def lookup(self, index):
            return self

        def start(self):
            return 0

        def end(self):
            return 1

    class MockValidator:
        def validate(self, value):
            raise ValidationError(text="required", code="required", index=["field"])

    token = MockToken(None)
    validator = MockValidator()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert len(error.messages()) == 1
        assert error.messages()[0].text == "The field 'field' is required."
        assert error.messages()[0].code == "required"
        assert error.messages()[0].index == ["field"]
        assert error.messages()[0].start_position == 0
        assert error.messages()[0].end_position == 1


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    class MockToken(Token):
        def _get_value(self):
            return {"key1": "value1"}
        def _get_child_token(self, key):
            return MockToken("value", 0, 5)
        def _get_key_token(self, key):
            return MockToken("key", 0, 3)
        def _get_position(self, index):
            return Position(1, 1, index)

    fields = {"key2": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 0)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'key2' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["key2"]
        assert messages[0].start_position == Position(1, 1, 0)
        assert messages[0].end_position == Position(1, 1, 0)


def test_validate_with_positions_invalid_key_error():
    class MockToken(Token):
        def _get_value(self):
            return {1: "value1"}
        def _get_child_token(self, key):
            return MockToken("value", 0, 5)
        def _get_key_token(self, key):
            return MockToken("key", 0, 3)
        def _get_position(self, index):
            return Position(1, 1, index)

    fields = {"key1": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 0)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "All object keys must be strings."
        assert messages[0].code == "invalid_key"
        assert messages[0].index == [1]
        assert messages[0].start_position == Position(1, 1, 0)
        assert messages[0].end_position == Position(1, 1, 0)


def test_validate_with_positions_success():
    class MockToken(Token):
        def _get_value(self):
            return {"key1": "value1"}
        def _get_child_token(self, key):
            return MockToken("value", 0, 5)
        def _get_key_token(self, key):
            return MockToken("key", 0, 3)
        def _get_position(self, index):
            return Position(1, 1, index)

    fields = {"key1": Field()}
    schema = Schema(fields)
    token = MockToken(None, 0, 0)
    validated_value = validate_with_positions(token=token, validator=schema)
    assert validated_value == {"key1": "value1"}


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions_required_field():
    class TestToken(Token):
        def _get_value(self) -> typing.Any:
            return {}

        def _get_child_token(self, key: typing.Any) -> "Token":
            return TestToken(value=None, start_index=0, end_index=0)

    token = TestToken(value={}, start_index=0, end_index=0)
    schema = Schema(fields={"field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'field' is required."
        assert messages[0].code == "required"
        assert messages[0].start_position == Position(line_no=1, column_no=1, char_index=0)
        assert messages[0].end_position == Position(line_no=1, column_no=1, char_index=0)

def test_validate_with_positions_invalid_type():
    class TestToken(Token):
        def _get_value(self) -> typing.Any:
            return "invalid"

        def _get_child_token(self, key: typing.Any) -> "Token":
            return TestToken(value=None, start_index=0, end_index=0)

    token = TestToken(value="invalid", start_index=0, end_index=0)
    field = Field()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].start_position == Position(line_no=1, column_no=1, char_index=0)
        assert messages[0].end_position == Position(line_no=1, column_no=1, char_index=0)

def test_validate_with_positions_null_value():
    class TestToken(Token):
        def _get_value(self) -> typing.Any:
            return None

        def _get_child_token(self, key: typing.Any) -> "Token":
            return TestToken(value=None, start_index=0, end_index=0)

    token = TestToken(value=None, start_index=0, end_index=0)
    field = Field(allow_null=False)
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "May not be null."
        assert messages[0].code == "null"
        assert messages[0].start_position == Position(line_no=1, column_no=1, char_index=0)
        assert messages[0].end_position == Position(line_no=1, column_no=1, char_index=0)

def test_validate_with_positions_valid_value():
    class TestToken(Token):
        def _get_value(self) -> typing.Any:
            return "valid"

        def _get_child_token(self, key: typing.Any) -> "Token":
            return TestToken(value=None, start_index=0, end_index=0)

    token = TestToken(value="valid", start_index=0, end_index=0)
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "valid"


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions_required_field():
    token = Token(value={"name": "John"}, start_index=0, end_index=20, content='{"name": "John"}')
    validator = Schema(fields={"age": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].code == "required"
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 20

def test_validate_with_positions_invalid_type():
    token = Token(value="invalid", start_index=0, end_index=7, content='"invalid"')
    validator = Field()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 7

def test_validate_with_positions_nested_validation_error():
    token = Token(value={"user": {"name": 123}}, start_index=0, end_index=25, content='{"user": {"name": 123}}')
    validator = Schema(fields={"user": Schema(fields={"name": Field()})})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].start_position.char_index == 10
        assert messages[0].end_position.char_index == 24

def test_validate_with_positions_success():
    token = Token(value={"name": "John"}, start_index=0, end_index=20, content='{"name": "John"}')
    validator = Schema(fields={"name": Field()})
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}

def test_validate_with_positions_null_value():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    validator = Field(allow_null=True)
    result = validate_with_positions(token=token, validator=validator)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions_validation_error():
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    validator = Schema(fields={"field": Field(allow_null=False)})
    try:
        validator.validate(token.value)
    except ValidationError as error:
        assert error.messages()[0].code == "null"


# LLM-generated content at query #13
#--------------------------

def test_validate_with_positions_required_field():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 10

def test_validate_with_positions_invalid_type():
    token = Token(value="not an object", start_index=0, end_index=12, content='"not an object"')
    schema = Schema(fields={})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == []
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 12

def test_validate_with_positions_nested_validation_error():
    token = Token(
        value={"user": {"age": "not a number"}},
        start_index=0,
        end_index=25,
        content='{"user": {"age": "not a number"}}'
    )
    schema = Schema(fields={"user": Schema(fields={"age": Field()})})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == ["user", "age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 25

def test_validate_with_positions_successful_validation():
    token = Token(
        value={"name": "John", "age": 30},
        start_index=0,
        end_index=20,
        content='{"name": "John", "age": 30}'
    )
    schema = Schema(fields={"name": Field(), "age": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    validator = Schema(fields={"age": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"
        assert message.index == ["age"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_invalid_type_error():
    token = Token(value={"age": "twenty"}, start_index=0, end_index=15, content='{"age": "twenty"}')
    validator = Schema(fields={"age": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "Must be an object."
        assert message.code == "type"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 15

def test_validate_with_positions_invalid_key_error():
    token = Token(value={1: "John"}, start_index=0, end_index=10, content='{1: "John"}')
    validator = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "All object keys must be strings."
        assert message.code == "invalid_key"
        assert message.index == [1]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

def test_validate_with_positions_valid_value():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    validator = Schema(fields={"name": Field()})
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    class MockField:
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Error", code="custom")])

    token = Token(value=None, start_index=0, end_index=0, content="")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        assert isinstance(error, ValidationError)


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions_required_field():
    token = Token(value={"name": "John"}, start_index=0, end_index=10, content='{"name": "John"}')
    validator = Schema(fields={"name": Field(), "age": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 10


def test_validate_with_positions_invalid_type():
    token = Token(value="not an object", start_index=0, end_index=12, content='"not an object"')
    validator = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == []
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 12


def test_validate_with_positions_nested_validation_error():
    token = Token(
        value={"user": {"age": "not a number"}},
        start_index=0,
        end_index=25,
        content='{"user": {"age": "not a number"}}'
    )
    validator = Schema(fields={"user": Schema(fields={"age": Field()})})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Must be an object."
        assert messages[0].code == "type"
        assert messages[0].index == ["user", "age"]
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 25


def test_validate_with_positions_successful_validation():
    token = Token(
        value={"name": "John", "age": 30},
        start_index=0,
        end_index=20,
        content='{"name": "John", "age": 30}'
    )
    validator = Schema(fields={"name": Field(), "age": Field()})
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_null_value_allowed():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    validator = Schema(fields={"name": Field()}, allow_null=True)
    result = validate_with_positions(token=token, validator=validator)
    assert result is None


def test_validate_with_positions_null_value_not_allowed():
    token = Token(value=None, start_index=0, end_index=4, content="null")
    validator = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "May not be null."
        assert messages[0].code == "null"
        assert messages[0].index == []
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 4


