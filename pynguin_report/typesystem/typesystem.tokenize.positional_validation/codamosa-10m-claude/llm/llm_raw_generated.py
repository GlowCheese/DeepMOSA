####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field, String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.tokenize import tokenize
    
    # Test 1: Valid token passes validation
    token = Token(value="test_value", token_type="text", start_position=0, end_position=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: Invalid token raises ValidationError
    token = Token(value="not_an_integer", token_type="text", start_position=0, end_position=14)
    field = Integer()
    with pytest.raises(ValidationError):
        validate_with_positions(token=token, validator=field)
    
    # Test 3: Required field validation error
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, token_type="text", start_position=0, end_position=5)
    schema = TestSchema()
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert any(msg.code == "required" for msg in messages)
    
    # Test 4: Messages are sorted by position
    class ComplexSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(value={}, token_type="text", start_position=0, end_position=10)
    schema = ComplexSchema()
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        positions = [msg.start_position for msg in messages if hasattr(msg, 'start_position')]
        assert positions == sorted(positions, key=lambda p: p.char_index if hasattr(p, 'char_index') else 0)
    
    # Test 5: Non-required field validation error preserves message text
    token = Token(value="invalid", token_type="text", start_position=5, end_position=12)
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code != "required"


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


class MockPosition:
    def __init__(self, char_index=0):
        self.char_index = char_index


class MockToken:
    def __init__(self, value, start=None, end=None):
        self.value = value
        self.start = start or MockPosition(0)
        self.end = end or MockPosition(len(str(value)))
        self._lookups = {}
    
    def lookup(self, index):
        if index in self._lookups:
            return self._lookups[index]
        return self


def test_validate_with_positions_valid_value():
    """Test validate_with_positions with valid input."""
    token = MockToken("42")
    validator = Integer()
    result = validate_with_positions(token=token, validator=validator)
    assert result == 42


def test_validate_with_positions_invalid_type():
    """Test validate_with_positions with invalid type."""
    token = MockToken("not_a_number")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field():
    """Test validate_with_positions with required field missing."""
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0
    assert any("required" in msg.text.lower() for msg in messages)


def test_validate_with_positions_messages_sorted_by_position():
    """Test that validation error messages are sorted by position."""
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_with_string_field():
    """Test validate_with_positions with valid string field."""
    token = MockToken("hello")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"


def test_validate_with_positions_preserves_message_code():
    """Test that message code is preserved in validation error."""
    token = MockToken("not_a_number")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    assert all(msg.code is not None for msg in messages)


def test_validate_with_positions_with_schema():
    """Test validate_with_positions with a schema validator."""
    class TestSchema(Schema):
        age = Integer()
    
    token = MockToken({"age": 25})
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"age": 25}


def test_validate_with_positions_invalid_schema():
    """Test validate_with_positions with invalid schema data."""
    class TestSchema(Schema):
        age = Integer()
    
    token = MockToken({"age": "not_an_int"})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0


# LLM-generated content at query #3
#--------------------------

def test_validate_with_positions():
    # Test successful validation with Field
    token = Token(value="test_value", start=None, end=None)
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"

    # Test successful validation with Schema
    schema = Schema(fields={})
    token = Token(value={}, start=None, end=None)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {}

    # Test validation error with required field
    class TestSchema(Schema):
        name = Field(allow_null=False)

    token = Token(value={"age": 25}, start=None, end=None)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 10})()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    assert len(exc_info.value.messages()) > 0
    messages = list(exc_info.value.messages())
    assert any(msg.code == "required" for msg in messages)

    # Test validation error message formatting
    error_messages = list(exc_info.value.messages())
    assert all(hasattr(msg, 'start_position') for msg in error_messages)
    assert all(hasattr(msg, 'end_position') for msg in error_messages)

    # Test messages are sorted by position
    assert error_messages == sorted(
        error_messages, 
        key=lambda m: m.start_position.char_index if hasattr(m.start_position, 'char_index') else 0
    )

    # Test validation error with non-required field error
    class StrictSchema(Schema):
        age = Field(allow_null=False)

    token = Token(value={"age": "invalid_int"}, start=None, end=None)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 15})()
    
    with pytest.raises(ValidationError):
        validate_with_positions(token=token, validator=StrictSchema())


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_field = Mock(spec=Field)
    mock_field.validate.return_value = "valid_value"
    
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_field)
    assert result == "valid_value"
    mock_field.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with "required" code
    mock_field = Mock(spec=Field)
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["parent", "field_name"]
    mock_message.text = "Field is required"
    
    error = ValidationError(messages=[mock_message])
    error.messages = Mock(return_value=[mock_message])
    mock_field.validate.side_effect = error
    
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_child_token = Mock(spec=Token)
    mock_child_token.start = Mock(char_index=0)
    mock_child_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_child_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_field)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 1
    assert "field_name" in messages[0].text
    assert messages[0].code == "required"


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-"required" code
    mock_field = Mock(spec=Field)
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ["field"]
    mock_message.text = "Invalid value"
    
    error = ValidationError(messages=[mock_message])
    error.messages = Mock(return_value=[mock_message])
    mock_field.validate.side_effect = error
    
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_child_token = Mock(spec=Token)
    mock_child_token.start = Mock(char_index=0)
    mock_child_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_child_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_field)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_field = Mock(spec=Field)
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "invalid"
    mock_message1.index = ["field1"]
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "invalid"
    mock_message2.index = ["field2"]
    mock_message2.text = "Error 2"
    
    error = ValidationError(messages=[mock_message1, mock_message2])
    error.messages = Mock(return_value=[mock_message1, mock_message2])
    mock_field.validate.side_effect = error
    
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_child_token1 = Mock(spec=Token)
    mock_child_token1.start = Mock(char_index=20)
    mock_child_token1.end = Mock(char_index=30)
    
    mock_child_token2 = Mock(spec=Token)
    mock_child_token2.start = Mock(char_index=5)
    mock_child_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_child_token1, mock_child_token2]
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_field)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 2
    # Should be sorted by char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


def test_validate_with_positions_with_schema():
    # Test validation with Schema validator
    mock_schema = Mock(spec=Schema)
    mock_schema.validate.return_value = {"field": "value"}
    
    mock_token = Mock(spec=Token)
    mock_token.value = {"field": "value"}
    
    result = validate_with_positions(token=mock_token, validator=mock_schema)
    assert result == {"field": "value"}
    mock_schema.validate.assert_called_once_with({"field": "value"})


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with non-required error code
    field = Integer()
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(10, 0, 10))
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(msg, Message) for msg in messages)
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test 4: ValidationError with required error code
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    schema = RequiredSchema()
    token = Token(value={"name": None}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(msg, Message) for msg in messages)
    
    # Test 5: Messages are sorted by start_position char_index
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        if len(messages) > 1:
            char_indices = [msg.start_position.char_index for msg in messages]
            assert char_indices == sorted(char_indices)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(line=1, char_index=0), end=Position(line=1, char_index=5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(
        value={"name": "John"},
        start=Position(line=1, char_index=0),
        end=Position(line=1, char_index=20)
    )
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(
        value={"name": None},
        start=Position(line=1, char_index=0),
        end=Position(line=1, char_index=10)
    )
    schema = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.start_position is not None for msg in messages)
        assert any(msg.end_position is not None for msg in messages)
    
    # Test 4: Integer field validation error
    token = Token(
        value="not_an_integer",
        start=Position(line=1, char_index=0),
        end=Position(line=1, char_index=14)
    )
    field = Integer()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for msg in messages:
            assert msg.start_position is not None
            assert msg.end_position is not None
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(
        value={},
        start=Position(line=1, char_index=0),
        end=Position(line=1, char_index=10)
    )
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test 1: Valid validation - no error raised
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test 2: Required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = {"nested": {}}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    nested_token = Mock(spec=Token)
    nested_token.start = Mock(char_index=5)
    nested_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = nested_token
    
    # Create a message with "required" code
    error_message = Message(
        text="This field is required.",
        code="required",
        index=["nested", "field"]
    )
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert messages[0].text == "The field 'field' is required."
    assert messages[0].start_position == nested_token.start
    assert messages[0].end_position == nested_token.end


def test_validate_with_positions_non_required_error():
    # Test 3: Non-required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = {"field": "invalid"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=20)
    
    error_token = Mock(spec=Token)
    error_token.start = Mock(char_index=10)
    error_token.end = Mock(char_index=25)
    mock_token.lookup.return_value = error_token
    
    error_message = Message(
        text="Invalid format.",
        code="invalid",
        index=["field"]
    )
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid format."
    assert messages[0].start_position == error_token.start
    assert messages[0].end_position == error_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "invalid", "field2": "invalid"}
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=30)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=15)
    
    def lookup_side_effect(index):
        if index == ["field1"]:
            return token1
        elif index == ["field2"]:
            return token2
        return mock_token
    
    mock_token.lookup.side_effect = lookup_side_effect
    
    error_message1 = Message(
        text="Error 1",
        code="error1",
        index=["field1"]
    )
    error_message2 = Message(
        text="Error 2",
        code="error2",
        index=["field2"]
    )
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position.char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


def test_validate_with_positions_with_schema():
    # Test 5: Validation with Schema validator
    mock_token = Mock(spec=Token)
    mock_token.value = {"key": "value"}
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.return_value = {"key": "validated_value"}
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == {"key": "validated_value"}
    mock_validator.validate.assert_called_once_with({"key": "value"})


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: Required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["nested", "field_name"]
    mock_message.text = "This field is required."
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = validation_error
    
    mock_token.lookup.return_value = mock_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"


def test_validate_with_positions_non_required_error():
    # Test case 3: Non-required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = "invalid"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid_type"
    mock_message.index = ["field"]
    mock_message.text = "Invalid type provided."
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = validation_error
    
    mock_token.lookup.return_value = mock_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].text == "Invalid type provided."
    assert messages[0].code == "invalid_type"
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test"
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "error1"
    mock_message1.index = ["field1"]
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "error2"
    mock_message2.index = ["field2"]
    mock_message2.text = "Error 2"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[mock_message1, mock_message2])
    mock_validator.validate.side_effect = validation_error
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=10)
    token1.end = Mock(char_index=20)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=8)
    
    mock_token.lookup.side_effect = [token2, token1]
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(line=0, char_index=0), end=Position(line=0, char_index=5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(line=0, char_index=0), end=Position(line=0, char_index=20))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with non-required error
    token = Token(value="not_a_number", start=Position(line=0, char_index=0), end=Position(line=0, char_index=12))
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        assert messages[0].start_position == token.start
        assert messages[0].end_position == token.end
    
    # Test 4: ValidationError with required error
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    token = Token(value={}, start=Position(line=0, char_index=0), end=Position(line=0, char_index=10))
    schema = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        assert "required" in messages[0].text.lower() or messages[0].code == "required"
    
    # Test 5: Messages are sorted by char_index
    token = Token(value="test", start=Position(line=0, char_index=0), end=Position(line=0, char_index=4))
    field = String(max_length=2)
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        if len(messages) > 1:
            for i in range(len(messages) - 1):
                assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation with Field
    token = Token(value="123", start=0, end=3)
    token.start = type('Position', (), {'char_index': 0})()
    token.end = type('Position', (), {'char_index': 3})()
    
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "123"


def test_validate_with_positions_field_validation_error():
    # Test validation error with Field
    from unittest.mock import Mock
    
    token = Mock()
    token.value = "invalid"
    token.start = Mock()
    token.start.char_index = 0
    token.end = Mock()
    token.end.char_index = 7
    token.lookup = Mock(return_value=token)
    
    field = Mock(spec=Field)
    error_message = Message(
        text="Invalid value",
        code="invalid",
        index=[]
    )
    field.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field():
    # Test required field validation error
    from unittest.mock import Mock
    
    token = Mock()
    token.value = {}
    token.start = Mock()
    token.start.char_index = 0
    token.end = Mock()
    token.end.char_index = 2
    token.lookup = Mock(return_value=token)
    
    schema = Mock(spec=Schema)
    error_message = Message(
        text="",
        code="required",
        index=["field_name"]
    )
    schema.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert "required" in [m.code for m in messages]
    assert any("field_name" in m.text for m in messages if "required" in m.code)


def test_validate_with_positions_multiple_errors():
    # Test multiple validation errors sorted by position
    from unittest.mock import Mock
    
    token = Mock()
    token.value = {}
    token.start = Mock()
    token.start.char_index = 0
    token.end = Mock()
    token.end.char_index = 10
    
    def lookup_side_effect(index):
        mock_token = Mock()
        mock_token.start = Mock()
        mock_token.start.char_index = len(index) * 2
        mock_token.end = Mock()
        mock_token.end.char_index = len(index) * 2 + 5
        return mock_token
    
    token.lookup = Mock(side_effect=lookup_side_effect)
    
    schema = Mock(spec=Schema)
    error_messages = [
        Message(text="", code="invalid", index=["field1"]),
        Message(text="", code="invalid", index=["field2"]),
    ]
    schema.validate.side_effect = ValidationError(messages=error_messages)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Check that messages are sorted by start_position
    for i in range(len(messages) - 1):
        assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_nested_schema():
    # Test validation with nested schema structure
    from unittest.mock import Mock
    
    token = Mock()
    token.value = {"nested": {}}
    token.start = Mock()
    token.start.char_index = 0
    token.end = Mock()
    token.end.char_index = 20
    
    def lookup_side_effect(index):
        mock_token = Mock()
        mock_token.start = Mock()
        mock_token.start.char_index = 5
        mock_token.end = Mock()
        mock_token.end.char_index = 15
        return mock_token
    
    token.lookup = Mock(side_effect=lookup_side_effect)
    
    schema = Mock(spec=Schema)
    error_message = Message(
        text="",
        code="required",
        index=["nested", "inner_field"]
    )
    schema.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].start_position.char_index == 5
    assert messages[0].end_position.char_index == 15


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation with Field
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="test_value", start=0, end=10)
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"


def test_validate_with_positions_field_validation_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="not_an_integer", start=0, end=14)
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    assert all(hasattr(m, 'start_position') for m in exc_info.value.messages())
    assert all(hasattr(m, 'end_position') for m in exc_info.value.messages())


def test_validate_with_positions_required_field():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    token = Token(value={}, start=0, end=5)
    validator = String(allow_null=False)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0


def test_validate_with_positions_schema_validation():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(value={"name": "John", "age": 30}, start=0, end=30)
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_sorted_by_position():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(value={}, start=0, end=10)
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    positions = [m.start_position.char_index for m in messages if hasattr(m.start_position, 'char_index')]
    assert positions == sorted(positions)


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from unittest.mock import Mock, MagicMock
    
    # Test 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=String)
    mock_validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "test_value"
    mock_validator.validate.assert_called_once_with("test_value")
    
    # Test 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    error_message = Message(
        text="Field is required",
        code="required",
        index=["name"]
    )
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with_position_token = Mock(spec=Token)
    with_position_token.start = Mock(char_index=0)
    with_position_token.end = Mock(char_index=5)
    mock_token.lookup.return_value = with_position_token
    
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert "name" in messages[0].text
        assert messages[0].code == "required"
        assert messages[0].start_position == with_position_token.start
        assert messages[0].end_position == with_position_token.end
    
    # Test 3: ValidationError with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"age": "invalid"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=20)
    
    error_message = Message(
        text="Not a valid integer",
        code="type_error",
        index=["age"]
    )
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=5)
    lookup_token.end = Mock(char_index=12)
    mock_token.lookup.return_value = lookup_token
    
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].text == "Not a valid integer"
        assert messages[0].code == "type_error"
        assert messages[0].start_position == lookup_token.start
        assert messages[0].end_position == lookup_token.end
    
    # Test 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "invalid", "field2": "invalid"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=50)
    
    error_message1 = Message(
        text="Error 1",
        code="error",
        index=["field1"]
    )
    error_message2 = Message(
        text="Error 2",
        code="error",
        index=["field2"]
    )
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(
        messages=[error_message1, error_message2]
    )
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=25)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=10)
    token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [token1, token2]
    
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 10
        assert messages[1].start_position.char_index == 20


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid token with Field validator
    token = Token(
        value="hello",
        start=Position(line=0, char_index=0),
        end=Position(line=0, char_index=5)
    )
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"
    
    # Test 2: Valid token with Schema validator
    class TestSchema(Schema):
        name = String()
    
    token = Token(
        value={"name": "John"},
        start=Position(line=0, char_index=0),
        end=Position(line=0, char_index=20)
    )
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}
    
    # Test 3: Invalid token raises ValidationError with positions
    token = Token(
        value="not_a_number",
        start=Position(line=0, char_index=0),
        end=Position(line=0, char_index=12)
    )
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test 4: Required field error handling
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    token = Token(
        value={},
        start=Position(line=0, char_index=0),
        end=Position(line=0, char_index=2)
    )
    validator = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(msg, Message) for msg in messages)
        # Messages should be sorted by position
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ("field1", "subfield")
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "subfield" in messages[0].text
    assert messages[0].start_position == lookup_token.start
    assert messages[0].end_position == lookup_token.end


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ("field1",)
    error_message.text = "Invalid value"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    lookup_token1 = Mock(spec=Token)
    lookup_token1.start = Mock(char_index=20)
    lookup_token1.end = Mock(char_index=30)
    
    lookup_token2 = Mock(spec=Token)
    lookup_token2.start = Mock(char_index=5)
    lookup_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [lookup_token1, lookup_token2]
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ("field1",)
    error_message1.text = "Error 1"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "invalid"
    error_message2.index = ("field2",)
    error_message2.text = "Error 2"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


class MockToken:
    """Mock Token class for testing"""
    def __init__(self, value, start_pos=0, end_pos=10):
        self.value = value
        self.start = MockPosition(start_pos)
        self.end = MockPosition(end_pos)
    
    def lookup(self, index):
        """Return self for simplicity in tests"""
        return self


class MockPosition:
    """Mock Position class for testing"""
    def __init__(self, char_index=0):
        self.char_index = char_index


class TestSchema(Schema):
    """Test schema for validation"""
    name = String(max_length=10)
    age = Integer()


def test_validate_with_positions_success():
    """Test successful validation"""
    token = MockToken({"name": "John", "age": 30})
    validator = TestSchema()
    
    result = validate_with_positions(token=token, validator=validator)
    
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_field_validation_error():
    """Test validation error with field validation"""
    token = MockToken({"name": "John" * 5, "age": 30})  # name too long
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field_error():
    """Test validation error for required field"""
    token = MockToken({"age": 30})  # missing required name
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert any("required" in msg.code for msg in messages)


def test_validate_with_positions_messages_sorted():
    """Test that error messages are sorted by position"""
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_message_has_positions():
    """Test that returned messages have position information"""
    token = MockToken({"age": "invalid"})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    for message in messages:
        assert hasattr(message, 'start_position')
        assert hasattr(message, 'end_position')
        assert message.start_position is not None
        assert message.end_position is not None


def test_validate_with_positions_with_field_validator():
    """Test validation with a single Field validator"""
    token = MockToken("not_a_number")
    validator = Integer()
    
    with pytest.raises(ValidationError):
        validate_with_positions(token=token, validator=validator)


def test_validate_with_positions_preserves_message_code():
    """Test that message codes are preserved"""
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert all(hasattr(msg, 'code') for msg in messages)


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test case 1: Successful validation with Field
    token = Token(value="test_value", start=0, end=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test case 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=20)
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test case 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=0, end=20)
    schema = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)
    
    # Test case 4: ValidationError with invalid type
    token = Token(value="not_an_integer", start=0, end=14)
    field = Integer()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None
    
    # Test case 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(value={}, start=0, end=50)
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        # Verify messages are sorted by start_position
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid token with Field validator
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"
    
    # Test 2: Valid token with Schema validator
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "test"}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "test"}
    
    # Test 3: Invalid value raises ValidationError with positional message
    token = Token(value="not_a_number", start=Position(0, 0, 0), end=Position(12, 0, 12))
    validator = Integer()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Required field error
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    validator = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
        assert any("required" in m.text.lower() or m.code == "required" for m in messages)
    
    # Test 5: Messages are sorted by start_position
    token = Token(value="test", start=Position(0, 0, 0), end=Position(4, 0, 4))
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


# LLM-generated content at query #18
#--------------------------

def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid validation passes through
    token = Token(value="test_value", start=Position(0, 0, 0), end=Position(10, 0, 10))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: ValidationError with required field
    class TestSchema(Schema):
        name = String(allow_null=False)
        age = Integer(allow_null=False)
    
    token = Token(value={"age": 25}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) > 0
    assert any(m.code == "required" for m in messages)
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 3: ValidationError messages are sorted by position
    token = Token(value={"age": "invalid"}, start=Position(0, 0, 0), end=Position(30, 0, 30))
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) > 0
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)
    
    # Test 4: ValidationError with non-required code
    token = Token(value={"name": 123, "age": "not_an_int"}, start=Position(0, 0, 0), end=Position(40, 0, 40))
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) > 0
    assert all(isinstance(m, Message) for m in messages)


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(
        value="test_value",
        start=Position(line=1, column=1, char_index=0),
        end=Position(line=1, column=11, char_index=10),
    )
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(
        value={"name": "John"},
        start=Position(line=1, column=1, char_index=0),
        end=Position(line=1, column=20, char_index=19),
    )
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with non-required code
    class StrictSchema(Schema):
        age = Integer()
    
    token = Token(
        value={"age": "not_an_int"},
        start=Position(line=1, column=1, char_index=0),
        end=Position(line=1, column=25, char_index=24),
    )
    schema = StrictSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        assert any(m.start_position is not None for m in messages)
        assert any(m.end_position is not None for m in messages)
    
    # Test 4: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        first = String(allow_null=False)
        second = String(allow_null=False)
    
    token = Token(
        value={},
        start=Position(line=1, column=1, char_index=0),
        end=Position(line=1, column=10, char_index=9),
    )
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 1
        # Verify messages are sorted by char_index
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    import pytest

    # Test 1: Successful validation with Field
    token = Token(value="hello", start=0, end=5)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"

    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=20)
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

    # Test 3: Validation error with Field
    field = Integer()
    token = Token(value="not_an_int", start=0, end=10)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    errors = exc_info.value.messages()
    assert len(errors) > 0
    assert all(hasattr(msg, 'start_position') for msg in errors)
    assert all(hasattr(msg, 'end_position') for msg in errors)

    # Test 4: Required field validation error
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=0, end=5)
    schema = RequiredSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    errors = exc_info.value.messages()
    assert len(errors) > 0
    assert any(msg.code == "required" for msg in errors)
    assert all(hasattr(msg, 'start_position') for msg in errors)

    # Test 5: Messages are sorted by start_position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={}, start=0, end=10)
    schema = MultiFieldSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    errors = exc_info.value.messages()
    if len(errors) > 1:
        for i in range(len(errors) - 1):
            assert errors[i].start_position.char_index <= errors[i + 1].start_position.char_index

    # Test 6: Token with custom positions
    token = Token(value="test", start=10, end=14)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import ValidationError, Message
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Valid token passes validation
    token = Token(value="test_string", start=0, end=11)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_string"
    
    # Test 2: Invalid token raises ValidationError with position info
    token = Token(value=123, start=0, end=3)
    field = String()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test 3: Schema validation with required field error
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=0, end=2)
    schema = TestSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        # Check that messages are sorted by position
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)
    
    # Test 4: Integer field validation
    token = Token(value=42, start=5, end=7)
    field = Integer()
    result = validate_with_positions(token=token, validator=field)
    assert result == 42
    
    # Test 5: Invalid integer value
    token = Token(value="not_an_int", start=0, end=10)
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 10
    
    # Test 6: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={}, start=0, end=5)
    schema = MultiFieldSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        # Verify messages are sorted by start_position
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid token passes validation
    token = Token(value="test_value", start=Position(0, 0, 0), end=Position(10, 0, 10))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: ValidationError with required field
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(value={"age": 25}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    schema = TestSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
    
    # Test 3: ValidationError with invalid value
    token = Token(value="not_an_integer", start=Position(0, 0, 0), end=Position(14, 0, 14))
    field = Integer()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Messages are sorted by position
    class ComplexSchema(Schema):
        field1 = String()
        field2 = String()
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(50, 0, 50))
    schema = ComplexSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)
    
    # Test 5: Valid complex schema passes
    token = Token(
        value={"field1": "value1", "field2": "value2"},
        start=Position(0, 0, 0),
        end=Position(50, 0, 50)
    )
    schema = ComplexSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


class MockPosition:
    def __init__(self, char_index=0):
        self.char_index = char_index


class MockToken:
    def __init__(self, value, start=None, end=None):
        self.value = value
        self.start = start or MockPosition(0)
        self.end = end or MockPosition(len(str(value)))
        self._children = {}
    
    def lookup(self, index):
        if not index:
            return self
        key = index[0]
        if key not in self._children:
            self._children[key] = MockToken(
                f"child_{key}",
                MockPosition(10),
                MockPosition(20)
            )
        child = self._children[key]
        if len(index) > 1:
            return child.lookup(index[1:])
        return child


def test_validate_with_positions_success():
    """Test successful validation without errors."""
    field = String()
    token = MockToken("valid_string")
    
    result = validate_with_positions(token=token, validator=field)
    assert result == "valid_string"


def test_validate_with_positions_validation_error_with_code():
    """Test validation error with non-required error code."""
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(
                messages=[
                    Message(
                        text="Invalid format",
                        code="invalid",
                        index=[]
                    )
                ]
            )
    
    field = FailingField()
    token = MockToken("invalid_value")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid format"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end


def test_validate_with_positions_required_error():
    """Test validation error with required error code."""
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(
                messages=[
                    Message(
                        text="",
                        code="required",
                        index=["field_name"]
                    )
                ]
            )
    
    field = FailingField()
    token = MockToken("some_value")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"


def test_validate_with_positions_nested_field_error():
    """Test validation error with nested field index."""
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(
                messages=[
                    Message(
                        text="Invalid nested value",
                        code="invalid",
                        index=["parent", "child"]
                    )
                ]
            )
    
    field = FailingField()
    token = MockToken("value")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid nested value"
    assert messages[0].index == ["parent", "child"]


def test_validate_with_positions_multiple_errors_sorted():
    """Test multiple validation errors are sorted by position."""
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(
                messages=[
                    Message(
                        text="Second error",
                        code="error2",
                        index=["field2"]
                    ),
                    Message(
                        text="First error",
                        code="error1",
                        index=["field1"]
                    ),
                ]
            )
    
    field = FailingField()
    token = MockToken("value")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by char_index
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


def test_validate_with_positions_empty_index():
    """Test validation error with empty index."""
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(
                messages=[
                    Message(
                        text="Root level error",
                        code="error",
                        index=[]
                    )
                ]
            )
    
    field = FailingField()
    token = MockToken("value")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Root level error"
    assert messages[0].start_position == token.start


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest
    
    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")
    
    # Test case 2: ValidationError with required field error
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["nested", "field"]
    mock_message.text = "Field is required"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages
    assert len(messages) == 1
    assert "field" in messages[0].text
    assert "required" in messages[0].text
    assert messages[0].code == "required"
    mock_token.lookup.assert_called_once_with(["nested"])
    
    # Test case 3: ValidationError with non-required error
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ["field"]
    mock_message.text = "Invalid value"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end
    
    # Test case 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_token_first = Mock(spec=Token)
    mock_token_first.start = Mock(char_index=10)
    mock_token_first.end = Mock(char_index=15)
    
    mock_token_second = Mock(spec=Token)
    mock_token_second.start = Mock(char_index=5)
    mock_token_second.end = Mock(char_index=8)
    
    mock_token.lookup.side_effect = [mock_token_second, mock_token_first]
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "invalid"
    mock_message1.index = ["field1"]
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "invalid"
    mock_message2.index = ["field2"]
    mock_message2.text = "Error 2"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message1, mock_message2]
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test case 1: Valid token passes validation
    token = Token(value="test_value", start=0, end=10)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 10})()
    
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test case 2: ValidationError with required field
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token_invalid = Token(value={}, start=0, end=5)
    token_invalid.start = type('obj', (object,), {'char_index': 0})()
    token_invalid.end = type('obj', (object,), {'char_index': 5})()
    token_invalid.lookup = lambda idx: token_invalid
    
    schema = TestSchema()
    
    try:
        validate_with_positions(token=token_invalid, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test case 3: ValidationError with non-required message
    token_error = Token(value="invalid", start=0, end=7)
    token_error.start = type('obj', (object,), {'char_index': 0})()
    token_error.end = type('obj', (object,), {'char_index': 7})()
    token_error.lookup = lambda idx: token_error
    
    field_int = Integer()
    
    try:
        validate_with_positions(token=token_error, validator=field_int)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test case 4: Messages are sorted by start_position
    token_sort = Token(value={}, start=0, end=10)
    token_sort.start = type('obj', (object,), {'char_index': 0})()
    token_sort.end = type('obj', (object,), {'char_index': 10})()
    token_sort.lookup = lambda idx: token_sort
    
    try:
        validate_with_positions(token=token_sort, validator=TestSchema())
    except ValidationError as error:
        messages = error.messages()
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field, String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from unittest.mock import Mock, MagicMock
    
    # Test 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "test_value"
    mock_validator.validate.assert_called_once_with("test_value")
    
    # Test 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=5)
    mock_token.lookup = Mock(return_value=mock_token)
    
    error_message = Message(
        text="Field is required",
        code="required",
        index=["name", "field"],
        start_position=Mock(char_index=0),
        end_position=Mock(char_index=5),
    )
    
    mock_validator = Mock(spec=Field)
    mock_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = mock_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) == 1
        assert "field" in messages[0].text.lower()
        assert messages[0].code == "required"
    
    assert with_error
    
    # Test 3: ValidationError with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"name": "invalid"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup = Mock(return_value=mock_token)
    
    error_message = Message(
        text="Invalid value",
        code="invalid",
        index=["name"],
        start_position=Mock(char_index=5),
        end_position=Mock(char_index=12),
    )
    
    mock_validator = Mock(spec=Field)
    mock_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = mock_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].text == "Invalid value"
        assert messages[0].code == "invalid"
    
    assert with_error
    
    # Test 4: Multiple ValidationErrors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "bad", "field2": "bad"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=20)
    mock_token.lookup = Mock(return_value=mock_token)
    
    error_message1 = Message(
        text="Error 1",
        code="error",
        index=["field1"],
        start_position=Mock(char_index=15),
        end_position=Mock(char_index=20),
    )
    error_message2 = Message(
        text="Error 2",
        code="error",
        index=["field2"],
        start_position=Mock(char_index=5),
        end_position=Mock(char_index=10),
    )
    
    mock_validator = Mock(spec=Field)
    mock_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = mock_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) == 2
        # Messages should be sorted by char_index
        assert messages[0].start_position.char_index == 5
        assert messages[1].start_position.char_index == 15
    
    assert with_error


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"name": "", "email": "test@example.com"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    error_message = Message(
        text="This field is required.",
        code="required",
        index=["name"]
    )
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "name" in messages[0].text
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"age": "invalid"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=20)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=8)
    mock_lookup_token.end = Mock(char_index=17)
    mock_token.lookup.return_value = mock_lookup_token
    
    error_message = Message(
        text="Not a valid integer.",
        code="type_error",
        index=["age"]
    )
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "type_error"
    assert messages[0].text == "Not a valid integer."
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "invalid", "field2": "invalid"}
    
    mock_lookup_token_1 = Mock(spec=Token)
    mock_lookup_token_1.start = Mock(char_index=20)
    mock_lookup_token_1.end = Mock(char_index=30)
    
    mock_lookup_token_2 = Mock(spec=Token)
    mock_lookup_token_2.start = Mock(char_index=5)
    mock_lookup_token_2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_lookup_token_1, mock_lookup_token_2]
    
    error_message_1 = Message(
        text="Error 1",
        code="error1",
        index=["field1"]
    )
    error_message_2 = Message(
        text="Error 2",
        code="error2",
        index=["field2"]
    )
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = ValidationError(
        messages=[error_message_1, error_message_2]
    )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Check that messages are sorted by char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


def test_validate_with_positions_nested_required_error():
    # Test required error with nested index
    mock_token = Mock(spec=Token)
    mock_token.value = {"user": {"name": ""}}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=30)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=10)
    mock_lookup_token.end = Mock(char_index=20)
    mock_token.lookup.return_value = mock_lookup_token
    
    error_message = Message(
        text="This field is required.",
        code="required",
        index=["user", "name"]
    )
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert "name" in messages[0].text
    mock_token.lookup.assert_called_once_with(["user"])


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_validation_error_required_field():
    # Test validation error with required field code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["field_name"]
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"


def test_validate_with_positions_validation_error_non_required():
    # Test validation error with non-required field code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=5)
    lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["nested", "field"]
    error_message.text = "Invalid value"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == lookup_token.start
    assert messages[0].end_position == lookup_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    lookup_token_1 = Mock(spec=Token)
    lookup_token_1.start = Mock(char_index=20)
    lookup_token_1.end = Mock(char_index=30)
    
    lookup_token_2 = Mock(spec=Token)
    lookup_token_2.start = Mock(char_index=5)
    lookup_token_2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [lookup_token_1, lookup_token_2]
    
    error_message_1 = Mock(spec=Message)
    error_message_1.code = "invalid"
    error_message_1.index = ["field1"]
    error_message_1.text = "Error 1"
    
    error_message_2 = Mock(spec=Message)
    error_message_2.code = "invalid"
    error_message_2.index = ["field2"]
    error_message_2.text = "Error 2"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message_1, error_message_2])
    validation_error.messages = Mock(return_value=[error_message_1, error_message_2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


def test_validate_with_positions():
    # Test successful validation
    mock_token = Token(value="test_value", start=Position(0, 0, 0), end=Position(0, 10, 10))
    field = String()
    result = validate_with_positions(token=mock_token, validator=field)
    assert result == "test_value"


def test_validate_with_positions_field_validation_error():
    # Test field validation error
    mock_token = Token(value="not_a_number", start=Position(0, 0, 0), end=Position(0, 12, 12))
    field = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_required_error():
    # Test required field validation error
    class TestSchema(Schema):
        name = String()
    
    mock_token = Token(value={}, start=Position(0, 0, 0), end=Position(0, 2, 2))
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    for message in messages:
        assert hasattr(message, 'start_position')
        assert hasattr(message, 'end_position')


def test_validate_with_positions_messages_sorted():
    # Test that messages are sorted by start_position
    class TestSchema(Schema):
        field1 = String()
        field2 = String()
    
    mock_token = Token(value={}, start=Position(0, 0, 0), end=Position(0, 2, 2))
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=schema)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_preserves_message_attributes():
    # Test that message attributes are preserved
    class TestSchema(Schema):
        required_field = String()
    
    mock_token = Token(value={}, start=Position(0, 0, 0), end=Position(0, 5, 5))
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=schema)
    
    messages = exc_info.value.messages()
    for message in messages:
        assert message.text is not None
        assert message.code is not None
        assert message.index is not None
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: Required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": None}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["nested", "field_name"]
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end


def test_validate_with_positions_non_required_error():
    # Test case 3: Non-required field validation error
    mock_token = Mock(spec=Token)
    mock_token.value = {"field": "invalid"}
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid_type"
    error_message.index = ["field"]
    error_message.text = "Invalid type"
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].text == "Invalid type"
    assert messages[0].code == "invalid_type"


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "invalid1", "field2": "invalid2"}
    
    token_1 = Mock(spec=Token)
    token_1.start = Mock(char_index=20)
    token_1.end = Mock(char_index=30)
    
    token_2 = Mock(spec=Token)
    token_2.start = Mock(char_index=5)
    token_2.end = Mock(char_index=10)
    
    mock_token.lookup.side_effect = [token_2, token_1]
    
    error_message_1 = Mock(spec=Message)
    error_message_1.code = "invalid_type"
    error_message_1.index = ["field1"]
    error_message_1.text = "Invalid type for field1"
    
    error_message_2 = Mock(spec=Message)
    error_message_2.code = "invalid_type"
    error_message_2.index = ["field2"]
    error_message_2.text = "Invalid type for field2"
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.side_effect = ValidationError(messages=[error_message_1, error_message_2])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position.char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Valid token passes validation
    token = Token(value="hello", start=0, end=5)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Invalid token with non-required field
    token = Token(value=12345, start=0, end=5)
    field = String(allow_null=True)
    result = validate_with_positions(token=token, validator=field)
    assert result == "12345"
    
    # Test 3: ValidationError with required field
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(value={}, start=0, end=10)
    with_error = False
    try:
        validate_with_positions(token=token, validator=TestSchema())
    except ValidationError as error:
        with_error = True
        messages = list(error.messages())
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)
    
    assert with_error is True
    
    # Test 4: ValidationError messages are sorted by position
    token = Token(value={"age": "invalid"}, start=0, end=20)
    with_error = False
    try:
        validate_with_positions(token=token, validator=TestSchema())
    except ValidationError as error:
        with_error = True
        messages = list(error.messages())
        # Verify messages are sorted by start_position
        positions = [msg.start_position.char_index for msg in messages if hasattr(msg.start_position, 'char_index')]
        assert positions == sorted(positions)
    
    assert with_error is True
    
    # Test 5: Valid integer passes validation
    token = Token(value=42, start=0, end=2)
    field = Integer()
    result = validate_with_positions(token=token, validator=field)
    assert result == 42


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema


def test_validate_with_positions():
    # Test case 1: Successful validation
    token = Mock()
    token.value = "test_value"
    validator = Mock(spec=Field)
    validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "validated_value"
    validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: ValidationError with required field message
    token = Mock()
    token.value = "test_value"
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup = Mock(return_value=token)
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["parent", "field_name"]
    error_message.text = "Field is required"
    
    validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end


def test_validate_with_positions_non_required_error():
    # Test case 3: ValidationError with non-required field message
    token = Mock()
    token.value = "test_value"
    token.start = Mock(char_index=5)
    token.end = Mock(char_index=15)
    token.lookup = Mock(return_value=token)
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field"]
    error_message.text = "Invalid value"
    
    validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple ValidationErrors sorted by position
    token = Mock()
    token.value = "test_value"
    token.lookup = Mock(return_value=token)
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ["field1"]
    error_message1.text = "Error 1"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "invalid"
    error_message2.index = ["field2"]
    error_message2.text = "Error 2"
    
    token1 = Mock()
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=25)
    
    token2 = Mock()
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=10)
    
    token.lookup.side_effect = [token1, token2]
    
    validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


def test_validate_with_positions_with_schema():
    # Test case 5: Validation with Schema object
    token = Mock()
    token.value = {"field": "value"}
    validator = Mock(spec=Schema)
    validator.validate.return_value = {"field": "validated_value"}
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"field": "validated_value"}
    validator.validate.assert_called_once_with({"field": "value"})


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=0, end=5)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=15)
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=0, end=10)
    schema = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(m, Message) for m in messages)
    
    # Test 4: Validation error with invalid type
    field = Integer()
    token = Token(value="not_an_integer", start=0, end=14)
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={}, start=0, end=10)
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)
    
    # Test 6: Validation error with non-required code
    field = String(max_length=5)
    token = Token(value="this_is_too_long", start=0, end=16)
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].text is not None


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="test_value", start=Position(line=1, char=0), end=Position(line=1, char=10))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(line=1, char=0), end=Position(line=1, char=20))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=Position(line=1, char=0), end=Position(line=1, char=15))
    schema = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') and hasattr(m, 'end_position') for m in messages)
    
    # Test 4: ValidationError with invalid type
    token = Token(value="not_an_int", start=Position(line=1, char=0), end=Position(line=1, char=10))
    field = Integer()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') and hasattr(m, 'end_position') for m in messages)
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={"field1": None, "field2": None}, start=Position(line=1, char=0), end=Position(line=1, char=30))
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        # Verify messages are sorted by char_index
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation with Field
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    field = String()
    token = Token(value="test_value", start=0, end=10)
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"


def test_validate_with_positions_field_validation_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    
    field = Integer()
    token = Token(value="not_an_integer", start=0, end=14)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_required_field():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    field = String(allow_null=False)
    token = Token(value=None, start=0, end=5)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert any(m.code == "required" for m in messages)


def test_validate_with_positions_schema():
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String()
    
    schema = TestSchema()
    token = Token(value={"name": "John"}, start=0, end=20)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_messages_sorted():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    schema = TestSchema()
    token = Token(value={}, start=0, end=10)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_error_text_preservation():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    field = String(max_length=5)
    token = Token(value="this_is_too_long", start=0, end=16)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(m.text for m in messages)


# LLM-generated content at query #36
#--------------------------

def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid token passes validation
    token = Token(value="test", start=Position(0, 0, 0), end=Position(4, 0, 4))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test 2: Integer validation succeeds
    token = Token(value="123", start=Position(0, 0, 0), end=Position(3, 0, 3))
    field = Integer()
    result = validate_with_positions(token=token, validator=field)
    assert result == 123
    
    # Test 3: Validation error with required field
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    schema = TestSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.code == "required" for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Validation error with invalid type
    token = Token(value="not_a_number", start=Position(0, 0, 0), end=Position(12, 0, 12))
    field = Integer()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    schema = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        # Verify messages are sorted by char_index
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    import pytest

    # Test 1: Successful validation with Field
    token = Token(value="test_value", start=0, end=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"

    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
        age = Integer()

    token = Token(value={"name": "John", "age": 30}, start=0, end=20)
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John", "age": 30}

    # Test 3: Validation error with Field
    token = Token(value=123, start=0, end=5)
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    assert len(exc_info.value.messages()) > 0

    # Test 4: Required field validation error
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)

    token = Token(value={}, start=0, end=10)
    schema = RequiredSchema()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)
    assert all(hasattr(msg, 'start_position') for msg in messages)
    assert all(hasattr(msg, 'end_position') for msg in messages)

    # Test 5: Messages are sorted by char_index
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)

    token = Token(value={}, start=0, end=20)
    schema = MultiFieldSchema()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    messages = exc_info.value.messages()
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest

    # Test case 1: Successful validation
    token = Mock()
    token.value = "test_value"
    
    validator = Mock(spec=Field)
    validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "validated_value"
    validator.validate.assert_called_once_with("test_value")

    # Test case 2: ValidationError with required field message
    token = Mock()
    token.value = "test_value"
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    
    nested_token = Mock()
    nested_token.start = Mock(char_index=0)
    nested_token.end = Mock(char_index=10)
    token.lookup.return_value = nested_token
    
    message = Mock(spec=Message)
    message.code = "required"
    message.index = ["field_name"]
    message.text = "Field is required"
    
    validator = Mock(spec=Field)
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    assert exc_info.value.messages()[0].code == "required"
    assert "field_name" in exc_info.value.messages()[0].text

    # Test case 3: ValidationError with non-required field message
    token = Mock()
    token.value = "invalid"
    token.start = Mock(char_index=5)
    token.end = Mock(char_index=12)
    
    nested_token = Mock()
    nested_token.start = Mock(char_index=5)
    nested_token.end = Mock(char_index=12)
    token.lookup.return_value = nested_token
    
    message = Mock(spec=Message)
    message.code = "invalid_type"
    message.index = ["field"]
    message.text = "Invalid type"
    
    validator = Mock(spec=Field)
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    assert exc_info.value.messages()[0].code == "invalid_type"
    assert exc_info.value.messages()[0].start_position.char_index == 5
    assert exc_info.value.messages()[0].end_position.char_index == 12

    # Test case 4: Multiple ValidationError messages sorted by position
    token = Mock()
    token.value = "test"
    
    msg1 = Mock(spec=Message)
    msg1.code = "error1"
    msg1.index = ["field1"]
    msg1.text = "Error 1"
    
    msg2 = Mock(spec=Message)
    msg2.code = "error2"
    msg2.index = ["field2"]
    msg2.text = "Error 2"
    
    token_lookup_1 = Mock()
    token_lookup_1.start = Mock(char_index=20)
    token_lookup_1.end = Mock(char_index=25)
    
    token_lookup_2 = Mock()
    token_lookup_2.start = Mock(char_index=5)
    token_lookup_2.end = Mock(char_index=10)
    
    token.lookup.side_effect = [token_lookup_2, token_lookup_1]
    
    validator = Mock(spec=Field)
    error = ValidationError(messages=[msg1, msg2])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index

    # Test case 5: ValidationError with Schema validator
    token = Mock()
    token.value = {"nested": "data"}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=15)
    
    nested_token = Mock()
    nested_token.start = Mock(char_index=0)
    nested_token.end = Mock(char_index=15)
    token.lookup.return_value = nested_token
    
    message = Mock(spec=Message)
    message.code = "required"
    message.index = ["nested_field"]
    message.text = "Nested field required"
    
    validator = Mock(spec=Schema)
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    assert exc_info.value.messages()[0].code == "required"


# LLM-generated content at query #39
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: ValidationError with 'required' code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["field_name"]
    error_message.text = "This field is required."
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"


def test_validate_with_positions_non_required_error():
    # Test case 3: ValidationError with non-'required' code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=5)
    lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["nested", "field"]
    error_message.text = "Invalid value."
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value."
    assert messages[0].code == "invalid"
    assert messages[0].start_position == lookup_token.start
    assert messages[0].end_position == lookup_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple ValidationErrors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    lookup_token_1 = Mock(spec=Token)
    lookup_token_1.start = Mock(char_index=20)
    lookup_token_1.end = Mock(char_index=25)
    
    lookup_token_2 = Mock(spec=Token)
    lookup_token_2.start = Mock(char_index=5)
    lookup_token_2.end = Mock(char_index=10)
    
    mock_token.lookup.side_effect = [lookup_token_1, lookup_token_2]
    
    error_message_1 = Mock(spec=Message)
    error_message_1.code = "invalid"
    error_message_1.index = ["field1"]
    error_message_1.text = "Error 1"
    
    error_message_2 = Mock(spec=Message)
    error_message_2.code = "type_error"
    error_message_2.index = ["field2"]
    error_message_2.text = "Error 2"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message_1, error_message_2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position.char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid validation passes through
    token = Token(value="hello", start=Position(line=0, char_index=0), end=Position(line=0, char_index=5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Invalid token raises ValidationError with positions
    token = Token(value=123, start=Position(line=0, char_index=0), end=Position(line=0, char_index=3))
    field = String()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 3: Schema validation with required field error
    class TestSchema(Schema):
        name = String()
        age = Integer(allow_null=False)
    
    token = Token(value={}, start=Position(line=0, char_index=0), end=Position(line=0, char_index=5))
    schema = TestSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for msg in messages:
            assert msg.start_position is not None
            assert msg.end_position is not None
    
    # Test 4: Messages are sorted by char_index
    token = Token(value="test", start=Position(line=0, char_index=0), end=Position(line=0, char_index=4))
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as error:
        messages = error.messages()
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)
    
    # Test 5: Valid integer passes through
    token = Token(value=42, start=Position(line=0, char_index=0), end=Position(line=0, char_index=2))
    field = Integer()
    result = validate_with_positions(token=token, validator=field)
    assert result == 42


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test case 1: Valid token with Field validator
    token = Token(value="hello", start=0, end=5)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 5})()
    
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test case 2: Valid token with Schema validator
    class TestSchema(Schema):
        name = String()
    
    schema = TestSchema()
    token = Token(value={"name": "John"}, start=0, end=20)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 20})()
    
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test case 3: Invalid token - non-integer value with Integer field
    token = Token(value="not_an_int", start=0, end=10)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 10})()
    token.lookup = lambda index: token
    
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test case 4: Required field validation error
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    schema = RequiredSchema()
    token = Token(value={}, start=0, end=10)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 10})()
    token.lookup = lambda index: token
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        # Messages should be sorted by char_index
        char_indices = [msg.start_position.char_index for msg in messages]
        assert char_indices == sorted(char_indices)


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


class Position:
    def __init__(self, char_index=0):
        self.char_index = char_index


class MockToken:
    def __init__(self, value, start_pos=0, end_pos=10):
        self.value = value
        self.start = Position(start_pos)
        self.end = Position(end_pos)
        self._children = {}
    
    def lookup(self, index):
        if not index:
            return self
        key = index[0]
        if key not in self._children:
            self._children[key] = MockToken(
                f"child_{key}", 
                start_pos=self.start.char_index + 5,
                end_pos=self.end.char_index + 5
            )
        if len(index) > 1:
            return self._children[key].lookup(index[1:])
        return self._children[key]


def test_validate_with_positions_valid_value():
    """Test with valid value that passes validation"""
    token = MockToken("test_value")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"


def test_validate_with_positions_validation_error_with_code():
    """Test validation error with non-required error code"""
    token = MockToken("invalid")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(msg, 'start_position') for msg in messages)
    assert all(hasattr(msg, 'end_position') for msg in messages)


def test_validate_with_positions_required_field_error():
    """Test validation error with required field missing"""
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert any('required' in msg.text.lower() for msg in messages)


def test_validate_with_positions_messages_sorted_by_position():
    """Test that messages are sorted by start position"""
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = MockToken({})
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_preserves_message_code():
    """Test that message code is preserved"""
    token = MockToken("invalid_int")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(msg.code is not None for msg in messages)


def test_validate_with_positions_nested_schema_error():
    """Test validation error in nested schema"""
    class InnerSchema(Schema):
        value = Integer()
    
    class OuterSchema(Schema):
        inner = InnerSchema()
    
    token = MockToken({"inner": {"value": "not_an_int"}})
    validator = OuterSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(msg, 'start_position') for msg in messages)


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_error():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ("field1", "subfield")
    mock_message.text = "Field is required"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "subfield" in messages[0].text
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end


def test_validate_with_positions_non_required_error():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ("field1",)
    mock_message.text = "Invalid value"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages
    assert len(messages) == 1
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_lookup_token1 = Mock(spec=Token)
    mock_lookup_token1.start = Mock(char_index=20)
    mock_lookup_token1.end = Mock(char_index=30)
    
    mock_lookup_token2 = Mock(spec=Token)
    mock_lookup_token2.start = Mock(char_index=5)
    mock_lookup_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_lookup_token1, mock_lookup_token2]
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "error1"
    mock_message1.index = ("field1",)
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "error2"
    mock_message2.index = ("field2",)
    mock_message2.text = "Error 2"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message1, mock_message2]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation
    token = Token(value="test_value", start=0, end=10)
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"


def test_validate_with_positions_with_schema():
    # Test successful validation with Schema
    token = Token(value={"name": "John"}, start=0, end=20)
    schema = Schema(fields={"name": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_required_error():
    # Test validation error with required field
    token = Token(value={}, start=0, end=10)
    token.lookup = lambda x: token
    
    schema = Schema(fields={"name": Field(allow_null=False)})
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_validation_error():
    # Test validation error with custom message
    token = Token(value="invalid", start=0, end=7)
    token.lookup = lambda x: token
    
    field = Field(max_length=5)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_multiple_errors():
    # Test multiple validation errors are sorted by position
    token = Token(value={}, start=0, end=100)
    token.lookup = lambda x: token
    
    schema = Schema(fields={
        "field1": Field(allow_null=False),
        "field2": Field(allow_null=False)
    })
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_error_with_required_code():
    # Test error message formatting for required field
    token = Token(value={}, start=0, end=10)
    token.lookup = lambda x: token
    
    schema = Schema(fields={"email": Field(allow_null=False)})
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert any("required" in str(m.code).lower() or "required" in m.text.lower() 
               for m in messages)


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with non-required code
    token = Token(value="not_a_number", start=Position(0, 0, 0), end=Position(12, 0, 12))
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(m, Message) for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: ValidationError with required code
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    token = Token(value={"required_field": None}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    schema = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any("required" in str(m.text).lower() or m.code == "required" for m in messages)
    
    # Test 5: Messages are sorted by start_position
    class MultiFieldSchema(Schema):
        field_a = String()
        field_b = String()
    
    token = Token(value={"field_a": 123, "field_b": 456}, start=Position(0, 0, 0), end=Position(30, 0, 30))
    schema = MultiFieldSchema()
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        if len(messages) > 1:
            for i in range(len(messages) - 1):
                assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions():
    # Test case 1: Valid validation passes through
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    field = String()
    token = Token(value="test", start=0, end=4)
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test case 2: ValidationError with required field
    from typesystem.fields import Integer
    
    field = Integer()
    token = Token(value="", start=0, end=0)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    errors = exc_info.value
    assert len(errors.messages()) > 0
    
    # Test case 3: ValidationError with non-required code
    from typesystem.fields import String
    
    field = String(max_length=5)
    token = Token(value="toolongstring", start=0, end=13)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    errors = exc_info.value
    messages = errors.messages()
    assert len(messages) > 0
    assert messages[0].start_position is not None
    assert messages[0].end_position is not None
    
    # Test case 4: Schema validation with required field error
    from typesystem.schemas import Schema
    from typesystem.fields import String
    
    class TestSchema(Schema):
        name = String()
    
    schema = TestSchema
    token = Token(value={}, start=0, end=2)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    errors = exc_info.value
    messages = errors.messages()
    assert len(messages) > 0
    
    # Test case 5: Messages are sorted by char_index
    field = String(max_length=5)
    token = Token(value="toolongstring", start=0, end=13)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    errors = exc_info.value
    messages = errors.messages()
    char_indices = [m.start_position.char_index for m in messages]
    assert char_indices == sorted(char_indices)


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest

    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")

    # Test case 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.text = "Field is required"
    mock_message.index = ["nested", "field"]
    
    error = ValidationError(messages=[mock_message])
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    assert len(exc_info.value.messages()) > 0
    raised_message = exc_info.value.messages()[0]
    assert raised_message.code == "required"
    assert "field" in raised_message.text
    assert raised_message.start_position == mock_lookup_token.start
    assert raised_message.end_position == mock_lookup_token.end

    # Test case 3: ValidationError with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.text = "Invalid value"
    mock_message.index = ["field"]
    
    error = ValidationError(messages=[mock_message])
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    assert len(exc_info.value.messages()) > 0
    raised_message = exc_info.value.messages()[0]
    assert raised_message.code == "invalid"
    assert raised_message.text == "Invalid value"
    assert raised_message.start_position == mock_lookup_token.start
    assert raised_message.end_position == mock_lookup_token.end

    # Test case 4: Multiple ValidationErrors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_lookup_token_1 = Mock(spec=Token)
    mock_lookup_token_1.start = Mock(char_index=10)
    mock_lookup_token_1.end = Mock(char_index=20)
    
    mock_lookup_token_2 = Mock(spec=Token)
    mock_lookup_token_2.start = Mock(char_index=0)
    mock_lookup_token_2.end = Mock(char_index=5)
    
    mock_token.lookup.side_effect = [mock_lookup_token_1, mock_lookup_token_2]
    
    mock_message_1 = Mock(spec=Message)
    mock_message_1.code = "invalid"
    mock_message_1.text = "Invalid 1"
    mock_message_1.index = ["field1"]
    
    mock_message_2 = Mock(spec=Message)
    mock_message_2.code = "invalid"
    mock_message_2.text = "Invalid 2"
    mock_message_2.index = ["field2"]
    
    error = ValidationError(messages=[mock_message_1, mock_message_2])
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 0
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from unittest.mock import Mock

    # Test 1: Valid validation passes through without error
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=String)
    mock_validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "test_value"
    mock_validator.validate.assert_called_once_with("test_value")

    # Test 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"name": "John"}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup = Mock(return_value=mock_token)
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["field_name"]
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text.lower()
    
    assert with_error

    # Test 3: ValidationError with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = {"age": "not_a_number"}
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=20)
    mock_token.lookup = Mock(return_value=mock_token)
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["age"]
    error_message.text = "Not a valid integer."
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) > 0
        assert messages[0].code == "invalid"
        assert messages[0].start_position == mock_token.start
        assert messages[0].end_position == mock_token.end
    
    assert with_error

    # Test 4: Multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {"field1": "invalid", "field2": "also_invalid"}
    
    mock_token_lookup_1 = Mock(spec=Token)
    mock_token_lookup_1.start = Mock(char_index=20)
    mock_token_lookup_1.end = Mock(char_index=30)
    
    mock_token_lookup_2 = Mock(spec=Token)
    mock_token_lookup_2.start = Mock(char_index=5)
    mock_token_lookup_2.end = Mock(char_index=15)
    
    def lookup_side_effect(index):
        if index == ["field1"]:
            return mock_token_lookup_1
        elif index == ["field2"]:
            return mock_token_lookup_2
        return mock_token
    
    mock_token.lookup = Mock(side_effect=lookup_side_effect)
    
    error_message_1 = Mock(spec=Message)
    error_message_1.code = "invalid"
    error_message_1.index = ["field1"]
    error_message_1.text = "Invalid field1"
    
    error_message_2 = Mock(spec=Message)
    error_message_2.code = "invalid"
    error_message_2.index = ["field2"]
    error_message_2.text = "Invalid field2"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message_1, error_message_2])
    mock_validator.validate.side_effect = validation_error
    
    with_error = False
    try:
        validate_with_positions(token=mock_token, validator=mock_validator)
    except ValidationError as e:
        with_error = True
        messages = e.messages()
        assert len(messages) == 2
        # Messages should be sorted by start_position char_index
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index
    
    assert with_error


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


class MockToken(Token):
    def __init__(self, value, start_pos=0, end_pos=10):
        self.value = value
        self.start = Position(char_index=start_pos, line_number=1, column_number=1)
        self.end = Position(char_index=end_pos, line_number=1, column_number=10)
        self._children = {}
    
    def lookup(self, index):
        if not index:
            return self
        token = self
        for idx in index:
            if idx not in token._children:
                token._children[idx] = MockToken(
                    f"child_{idx}",
                    start_pos=token.start.char_index + 5,
                    end_pos=token.end.char_index + 5
                )
            token = token._children[idx]
        return token


def test_validate_with_positions_success():
    """Test successful validation returns the validated value."""
    token = MockToken("123")
    field = Integer()
    result = validate_with_positions(token=token, validator=field)
    assert result == 123


def test_validate_with_positions_required_field_error():
    """Test validation error for required field."""
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = MockToken({})
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0
    assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_invalid_type_error():
    """Test validation error for invalid type."""
    token = MockToken("not_a_number")
    field = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0
    assert messages[0].start_position is not None
    assert messages[0].end_position is not None


def test_validate_with_positions_messages_sorted_by_position():
    """Test that error messages are sorted by start position."""
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = MockToken({})
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_message_attributes():
    """Test that positional messages have correct attributes."""
    token = MockToken("invalid")
    field = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = list(exc_info.value.messages())
    message = messages[0]
    assert hasattr(message, 'text')
    assert hasattr(message, 'code')
    assert hasattr(message, 'index')
    assert hasattr(message, 'start_position')
    assert hasattr(message, 'end_position')
    assert message.start_position is not None
    assert message.end_position is not None


def test_validate_with_positions_nested_schema_error():
    """Test validation error in nested schema."""
    class InnerSchema(Schema):
        value = Integer()
    
    class OuterSchema(Schema):
        inner = InnerSchema()
    
    token = MockToken({"inner": {"value": "not_int"}})
    schema = OuterSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation with Field
    token = Token(value="test_value", start=0, end=10)
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: ValidationError with required field
    token = Token(value=None, start=0, end=5)
    field = Field(required=True)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end
    
    # Test 3: ValidationError with custom message
    token = Token(value="invalid", start=0, end=7)
    field = Field(validator=lambda x: x != "invalid")
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end
    
    # Test 4: ValidationError with Schema
    schema = Schema(fields={"name": Field(required=True)})
    token = Token(value={}, start=0, end=10)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    
    # Test 5: Messages are sorted by position
    token = Token(value={"a": None, "b": None}, start=0, end=20)
    schema = Schema(fields={"a": Field(required=True), "b": Field(required=True)})
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    schema = TestSchema()
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with non-required code
    field = Integer()
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(10, 0, 10))
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Validation error with required code
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    schema = RequiredSchema()
    token = Token(value={}, start=Position(0, 0, 0), end=Position(5, 0, 5))
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.code == "required" for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    schema = MultiFieldSchema()
    token = Token(value={}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Valid token passes validation
    token = Token(value="test_value", start=0, end=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: ValidationError with required field message
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"age": 25}, start=0, end=20)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) > 0
    assert any(m.code == "required" for m in messages)
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 3: ValidationError with non-required field message
    field = Integer()
    token = Token(value="not_an_integer", start=5, end=20)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Messages are sorted by start_position
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)
    
    # Test 5: Valid schema validation
    class ValidSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=20)
    result = validate_with_positions(token=token, validator=ValidSchema())
    assert result == {"name": "John"}


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation without errors
    token = Mock(spec=Token)
    token.value = "test_value"
    validator = Mock(spec=Field)
    validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"
    validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test case 2: Validation error with "required" code
    token = Mock(spec=Token)
    token.value = {"field": "value"}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=5)
    lookup_token.end = Mock(char_index=15)
    token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.text = "This field is required."
    error_message.index = ["nested", "field"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert "field" in messages[0].text


def test_validate_with_positions_with_non_required_error():
    # Test case 3: Validation error with non-"required" code
    token = Mock(spec=Token)
    token.value = "invalid"
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=7)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=7)
    token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid_choice"
    error_message.text = "Not a valid choice."
    error_message.index = []
    
    validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "invalid_choice"
    assert messages[0].text == "Not a valid choice."


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    token = Mock(spec=Token)
    token.value = {"field1": "val1", "field2": "val2"}
    
    lookup_token1 = Mock(spec=Token)
    lookup_token1.start = Mock(char_index=20)
    lookup_token1.end = Mock(char_index=30)
    
    lookup_token2 = Mock(spec=Token)
    lookup_token2.start = Mock(char_index=10)
    lookup_token2.end = Mock(char_index=15)
    
    token.lookup.side_effect = [lookup_token2, lookup_token1]
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.text = "Invalid value 1"
    error_message1.index = ["field1"]
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "invalid"
    error_message2.text = "Invalid value 2"
    error_message2.index = ["field2"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    validation_error.messages = Mock(return_value=[error_message1, error_message2])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 10
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: ValidationError with required field message
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Field)
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["nested", "field"]
    error_message.text = "Field is required"
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    nested_token = Mock(spec=Token)
    nested_token.start = Mock(char_index=5)
    nested_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = nested_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field' is required."
    assert messages[0].code == "required"
    assert messages[0].start_position == nested_token.start
    assert messages[0].end_position == nested_token.end
    mock_token.lookup.assert_called_once_with(["nested"])


def test_validate_with_positions_non_required_error():
    # Test case 3: ValidationError with non-required error message
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Schema)
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field"]
    error_message.text = "Invalid value"
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    error_token = Mock(spec=Token)
    error_token.start = Mock(char_index=2)
    error_token.end = Mock(char_index=8)
    mock_token.lookup.return_value = error_token
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == error_token.start
    assert messages[0].end_position == error_token.end
    mock_token.lookup.assert_called_once_with(["field"])


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=Schema)
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ["field1"]
    error_message1.text = "Error 1"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "invalid"
    error_message2.index = ["field2"]
    error_message2.text = "Error 2"
    
    validation_error = ValidationError(messages=[error_message2, error_message1])
    mock_validator.validate.side_effect = validation_error
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=10)
    token1.end = Mock(char_index=15)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=8)
    
    mock_token.lookup.side_effect = [token2, token1]
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest
    
    # Test 1: Successful validation - no exception
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")
    
    # Test 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["field1", "subfield"]
    mock_message.text = "Field is required"
    mock_message.start = Mock(char_index=0)
    mock_message.end = Mock(char_index=10)
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    assert len(raised_error.messages) == 1
    assert raised_error.messages[0].code == "required"
    assert "subfield" in raised_error.messages[0].text
    assert raised_error.messages[0].start_position == mock_token.start
    
    # Test 3: ValidationError with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ["field1"]
    mock_message.text = "Invalid value"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    assert len(raised_error.messages) == 1
    assert raised_error.messages[0].code == "invalid"
    assert raised_error.messages[0].text == "Invalid value"
    
    # Test 4: Multiple validation errors sorted by position
    mock_token_1 = Mock(spec=Token)
    mock_token_1.start = Mock(char_index=10)
    mock_token_1.end = Mock(char_index=20)
    
    mock_token_2 = Mock(spec=Token)
    mock_token_2.start = Mock(char_index=5)
    mock_token_2.end = Mock(char_index=8)
    
    mock_token_root = Mock(spec=Token)
    mock_token_root.value = "test_value"
    mock_token_root.lookup.side_effect = lambda idx: mock_token_2 if idx == ["field2"] else mock_token_1
    
    mock_message_1 = Mock(spec=Message)
    mock_message_1.code = "invalid"
    mock_message_1.index = ["field1"]
    mock_message_1.text = "Error 1"
    
    mock_message_2 = Mock(spec=Message)
    mock_message_2.code = "invalid"
    mock_message_2.index = ["field2"]
    mock_message_2.text = "Error 2"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message_1, mock_message_2]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token_root, validator=mock_validator)
    
    raised_error = exc_info.value
    assert len(raised_error.messages) == 2
    # Messages should be sorted by char_index
    assert raised_error.messages[0].start_position.char_index == 5
    assert raised_error.messages[1].start_position.char_index == 10


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["parent", "field"]
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "field" in messages[0].text
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field"]
    error_message.text = "Invalid value"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 1
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"
    assert messages[0].start_position == mock_token.start


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by char_index
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    token_first = Mock(spec=Token)
    token_first.start = Mock(char_index=0)
    token_first.end = Mock(char_index=5)
    
    token_second = Mock(spec=Token)
    token_second.start = Mock(char_index=10)
    token_second.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [token_second, token_first]
    
    error_message_1 = Mock(spec=Message)
    error_message_1.code = "invalid"
    error_message_1.index = ["field2"]
    error_message_1.text = "Invalid value 2"
    
    error_message_2 = Mock(spec=Message)
    error_message_2.code = "type_error"
    error_message_2.index = ["field1"]
    error_message_2.text = "Invalid type 1"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message_1, error_message_2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 0
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from unittest.mock import Mock, MagicMock
    
    # Test case 1: Successful validation
    token = Mock(spec=Token)
    token.value = "test_value"
    
    validator = Mock(spec=String)
    validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"
    validator.validate.assert_called_once_with("test_value")
    
    # Test case 2: ValidationError with required field message
    token = Mock(spec=Token)
    token.value = {"name": ""}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup.return_value = token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.text = "Field is required"
    error_message.index = ["name"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    
    # Test case 3: ValidationError with non-required field message
    token = Mock(spec=Token)
    token.value = {"age": "invalid"}
    token.start = Mock(char_index=5)
    token.end = Mock(char_index=15)
    token.lookup.return_value = token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.text = "Not a valid integer"
    error_message.index = ["age"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "invalid"
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end
    
    # Test case 4: ValidationError with multiple messages sorted by position
    token = Mock(spec=Token)
    token.value = {"field1": "", "field2": ""}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup.return_value = token
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "required"
    error_message1.text = "Field1 is required"
    error_message1.index = ["field1"]
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "required"
    error_message2.text = "Field2 is required"
    error_message2.index = ["field2"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message2, error_message1])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Verify messages are sorted by char_index
    for i in range(len(messages) - 1):
        assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation
    token = Mock(spec=Token)
    token.value = "test_value"
    validator = Mock(spec=Field)
    validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "validated_value"
    validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test case 2: Required field validation error
    token = Mock(spec=Token)
    token.value = None
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=5)
    token.lookup.return_value = token
    
    validator = Mock(spec=Field)
    message = Mock(spec=Message)
    message.code = "required"
    message.index = ["field_name"]
    message.text = "This field is required."
    
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert "required" in [m.code for m in messages]


def test_validate_with_positions_with_validation_error():
    # Test case 3: Regular validation error
    token = Mock(spec=Token)
    token.value = "invalid"
    token.start = Mock(char_index=10)
    token.end = Mock(char_index=17)
    token.lookup.return_value = token
    
    validator = Mock(spec=Field)
    message = Mock(spec=Message)
    message.code = "invalid_type"
    message.index = []
    message.text = "Invalid value type."
    
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    token = Mock(spec=Token)
    token.value = {"field1": None, "field2": None}
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=25)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=10)
    token2.end = Mock(char_index=15)
    
    token.lookup.side_effect = [token2, token1]
    
    validator = Mock(spec=Schema)
    message1 = Mock(spec=Message)
    message1.code = "required"
    message1.index = ["field1"]
    message1.text = "Field1 is required."
    
    message2 = Mock(spec=Message)
    message2.code = "required"
    message2.index = ["field2"]
    message2.text = "Field2 is required."
    
    error = ValidationError(messages=[message1, message2])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Verify messages are sorted by char_index
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


def test_validate_with_positions_nested_index_error():
    # Test case 5: Nested field validation error
    token = Mock(spec=Token)
    token.value = {"nested": {"field": None}}
    nested_token = Mock(spec=Token)
    nested_token.start = Mock(char_index=5)
    nested_token.end = Mock(char_index=10)
    token.lookup.return_value = nested_token
    
    validator = Mock(spec=Schema)
    message = Mock(spec=Message)
    message.code = "invalid"
    message.index = ["nested", "field"]
    message.text = "Invalid nested field."
    
    error = ValidationError(messages=[message])
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    token.lookup.assert_called_with(["nested", "field"])


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positions import Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with Field
    field = Integer()
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(10, 0, 10))
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Validation error with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    schema = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for msg in messages:
            assert hasattr(msg, 'start_position')
            assert hasattr(msg, 'end_position')
            assert msg.start_position is not None
            assert msg.end_position is not None
    
    # Test 5: Messages are sorted by start_position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    schema = MultiFieldSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test case 1: Successful validation without errors
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test case 2: ValidationError with required field code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Field)
    
    # Create error message with required code
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["nested", "field_name"]
    error_message.text = "This field is required"
    
    # Mock the lookup method
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) > 0
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"
    mock_token.lookup.assert_called_once_with(["nested"])


def test_validate_with_positions_validation_error():
    # Test case 3: ValidationError with non-required code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Field)
    
    # Create error message with invalid code
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field"]
    error_message.text = "Invalid value"
    
    # Mock the lookup method
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=2)
    mock_lookup_token.end = Mock(char_index=8)
    mock_token.lookup.return_value = mock_lookup_token
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) > 0
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end
    mock_token.lookup.assert_called_once_with(["field"])


def test_validate_with_positions_multiple_errors_sorted():
    # Test case 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_validator = Mock(spec=Schema)
    
    # Create multiple error messages
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ["field1"]
    error_message1.text = "Error 1"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "required"
    error_message2.index = ["field2", "nested"]
    error_message2.text = "Error 2"
    
    # Create lookup tokens with different positions
    mock_lookup_token1 = Mock(spec=Token)
    mock_lookup_token1.start = Mock(char_index=10)
    mock_lookup_token1.end = Mock(char_index=20)
    
    mock_lookup_token2 = Mock(spec=Token)
    mock_lookup_token2.start = Mock(char_index=5)
    mock_lookup_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_lookup_token1, mock_lookup_token2]
    
    validation_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 2
    # Messages should be sorted by char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 10


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation with Field
    token = Token(value="test", start=0, end=4)
    field = Field()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"


def test_validate_with_positions_field_validation_error():
    # Test validation error with Field
    token = Token(value="", start=0, end=0)
    field = Field(allow_null=False)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field_error():
    # Test required field validation error
    schema = Schema(fields={"name": Field(allow_null=False)})
    token = Token(value={}, start=0, end=5)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_error_message_positions():
    # Test that error messages have correct position information
    token = Token(value={"age": "invalid"}, start=0, end=20)
    schema = Schema(fields={"age": Field()})
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    assert all(hasattr(msg, 'start_position') for msg in messages)
    assert all(hasattr(msg, 'end_position') for msg in messages)


def test_validate_with_positions_sorted_messages():
    # Test that error messages are sorted by position
    schema = Schema(fields={
        "field1": Field(allow_null=False),
        "field2": Field(allow_null=False)
    })
    token = Token(value={}, start=0, end=100)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    positions = [msg.start_position.char_index for msg in messages if hasattr(msg.start_position, 'char_index')]
    assert positions == sorted(positions)


def test_validate_with_positions_with_schema():
    # Test successful validation with Schema
    schema = Schema(fields={"name": Field()})
    token = Token(value={"name": "John"}, start=0, end=20)
    
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_preserves_message_code():
    # Test that original message codes are preserved
    field = Field(choices=["a", "b"])
    token = Token(value="c", start=0, end=1)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = list(exc_info.value.messages())
    assert any(msg.code == "choice" for msg in messages)


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    schema = TestSchema()
    token = Token(value={"name": "test"}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}
    
    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    schema = RequiredSchema()
    token = Token(value={"name": None}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.start_position is not None for m in messages)
        assert any(m.end_position is not None for m in messages)
    
    # Test 4: ValidationError with non-required code
    field = Integer()
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(10, 0, 10))
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for message in messages:
            assert message.start_position is not None
            assert message.end_position is not None
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    schema = MultiFieldSchema()
    token = Token(value={}, start=Position(0, 0, 0), end=Position(20, 0, 20))
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        if len(messages) > 1:
            for i in range(len(messages) - 1):
                assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    
    # Test 1: Successful validation with Field
    token = Token(value="test_value", start=0, end=10)
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=20)
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=0, end=20)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 20})()
    
    validator = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
    
    # Test 4: ValidationError with non-required error code
    token = Token(value="not_an_integer", start=0, end=14)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 14})()
    
    validator = Integer()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        # Verify messages are sorted by position
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)
    
    # Test 5: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = Integer(allow_null=False)
    
    token = Token(value={"field1": None, "field2": "invalid"}, start=0, end=30)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 30})()
    
    validator = MultiFieldSchema()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        if len(messages) > 1:
            positions = [m.start_position.char_index for m in messages]
            assert positions == sorted(positions)


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_with_positions():
    # Test with valid token and field validator
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="hello", start=0, end=5)
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"


def test_validate_with_positions_with_schema():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(value={"name": "John", "age": 30}, start=0, end=20)
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_validation_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="not_a_number", start=0, end=12)
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    for message in exc_info.value.messages():
        assert hasattr(message, 'start_position')
        assert hasattr(message, 'end_position')


def test_validate_with_positions_required_field_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=0, end=10)
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    for message in messages:
        assert message.start_position is not None
        assert message.end_position is not None


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(value={}, start=0, end=10)
    validator = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_with_nested_schema():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class AddressSchema(Schema):
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema()
    
    token = Token(value={"name": "John", "address": {"city": "NYC"}}, start=0, end=30)
    validator = PersonSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result["name"] == "John"
    assert result["address"]["city"] == "NYC"


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_with_positions():
    # Test successful validation with Field
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="test_value", start=0, end=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"


def test_validate_with_positions_field_validation_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    
    token = Token(value="not_an_int", start=0, end=10)
    field = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field():
    from typesystem.fields import String, Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=0, end=10)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_with_schema():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(value={"name": "John", "age": 30}, start=0, end=50)
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_message_sorting():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    class TestSchema(Schema):
        field1 = Integer()
        field2 = Integer()
    
    token = Token(value={"field1": "invalid", "field2": "also_invalid"}, start=0, end=100)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    
    messages = exc_info.value.messages()
    # Verify messages are sorted by start_position
    positions = [msg.start_position.char_index for msg in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_preserves_message_text():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    
    token = Token(value=123, start=0, end=10)
    field = String()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert all(hasattr(msg, 'text') for msg in messages)
    assert all(hasattr(msg, 'code') for msg in messages)
    assert all(hasattr(msg, 'start_position') for msg in messages)
    assert all(hasattr(msg, 'end_position') for msg in messages)


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


class MockToken:
    def __init__(self, value, start_pos=0, end_pos=10):
        self.value = value
        self.start = Position(char_index=start_pos, line_number=1, column_number=start_pos)
        self.end = Position(char_index=end_pos, line_number=1, column_number=end_pos)
        self._lookup_result = None

    def lookup(self, index):
        if self._lookup_result:
            return self._lookup_result
        return self


class TestSchema(Schema):
    name = String(max_length=10)
    age = Integer()


def test_validate_with_positions_success():
    """Test successful validation returns the validated value."""
    token = MockToken("valid_string")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid_string"


def test_validate_with_positions_field_validation_error():
    """Test validation error with Field validator."""
    token = MockToken("x" * 100)
    validator = String(max_length=10)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_required_field_error():
    """Test validation error with required field missing."""
    token = MockToken({})
    
    class SimpleSchema(Schema):
        name = String(allow_null=False)
    
    validator = SimpleSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    # Required field error should have custom message format
    assert any("required" in str(m.code).lower() or "required" in m.text.lower() for m in messages)


def test_validate_with_positions_error_messages_sorted():
    """Test that error messages are sorted by character position."""
    token = MockToken({})
    
    class MultiFieldSchema(Schema):
        field_a = String(allow_null=False)
        field_b = Integer(allow_null=False)
    
    validator = MultiFieldSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


def test_validate_with_positions_integer_validation():
    """Test validation error with Integer field."""
    token = MockToken("not_an_integer")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)


def test_validate_with_positions_message_attributes():
    """Test that positional messages have all required attributes."""
    token = MockToken("invalid")
    validator = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    for message in messages:
        assert hasattr(message, 'text')
        assert hasattr(message, 'code')
        assert hasattr(message, 'index')
        assert hasattr(message, 'start_position')
        assert hasattr(message, 'end_position')


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_result"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_result"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=0)
    lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["field1", "subfield"]
    error_message.text = "Field is required"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "subfield" in messages[0].text
    assert messages[0].start_position == lookup_token.start
    assert messages[0].end_position == lookup_token.end


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    
    lookup_token = Mock(spec=Token)
    lookup_token.start = Mock(char_index=5)
    lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = lookup_token
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field1"]
    error_message.text = "Invalid value"
    
    mock_validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"
    assert messages[0].start_position == lookup_token.start
    assert messages[0].end_position == lookup_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    lookup_token1 = Mock(spec=Token)
    lookup_token1.start = Mock(char_index=20)
    lookup_token1.end = Mock(char_index=25)
    
    lookup_token2 = Mock(spec=Token)
    lookup_token2.start = Mock(char_index=5)
    lookup_token2.end = Mock(char_index=10)
    
    mock_token.lookup.side_effect = [lookup_token2, lookup_token1]
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ["field1"]
    error_message1.text = "First error"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "type_error"
    error_message2.index = ["field2"]
    error_message2.text = "Second error"
    
    mock_validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "test_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "test_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_validation_error():
    # Test validation error handling
    mock_token = Mock(spec=Token)
    mock_token.value = "invalid_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_validator = Mock(spec=Field)
    
    # Create mock error messages
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.text = "Invalid value"
    mock_message.index = ["field"]
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    assert len(exc_info.value.messages) == 1
    assert exc_info.value.messages[0].text == "Invalid value"
    assert exc_info.value.messages[0].start_position == mock_token.start
    assert exc_info.value.messages[0].end_position == mock_token.end


def test_validate_with_positions_required_field_error():
    # Test required field error handling
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_validator = Mock(spec=Schema)
    
    # Create mock error message for required field
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.text = "This field is required."
    mock_message.index = ["nested", "field"]
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    assert len(exc_info.value.messages) == 1
    assert "field" in exc_info.value.messages[0].text
    assert "required" in exc_info.value.messages[0].text


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    
    mock_lookup_token_1 = Mock(spec=Token)
    mock_lookup_token_1.start = Mock(char_index=20)
    mock_lookup_token_1.end = Mock(char_index=30)
    
    mock_lookup_token_2 = Mock(spec=Token)
    mock_lookup_token_2.start = Mock(char_index=10)
    mock_lookup_token_2.end = Mock(char_index=20)
    
    mock_token.lookup.side_effect = [mock_lookup_token_1, mock_lookup_token_2]
    
    mock_validator = Mock(spec=Schema)
    
    # Create multiple mock error messages
    mock_message_1 = Mock(spec=Message)
    mock_message_1.code = "invalid"
    mock_message_1.text = "Error 1"
    mock_message_1.index = ["field1"]
    
    mock_message_2 = Mock(spec=Message)
    mock_message_2.code = "invalid"
    mock_message_2.text = "Error 2"
    mock_message_2.index = ["field2"]
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message_1, mock_message_2]
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 10
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ("field1", "field2")
    mock_message.text = "Field is required"
    
    mock_validator = Mock(spec=Schema)
    error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field2' is required."
    assert messages[0].code == "required"
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-required field
    mock_token = Mock(spec=Token)
    mock_token.value = "invalid"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid_type"
    mock_message.index = ("field1",)
    mock_message.text = "Invalid type provided"
    
    mock_validator = Mock(spec=Field)
    error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid type provided"
    assert messages[0].code == "invalid_type"


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=25)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=10)
    
    mock_token.lookup.side_effect = [token2, token1]
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "invalid"
    mock_message1.index = ("field1",)
    mock_message1.text = "Invalid field1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "required"
    mock_message2.index = ("field2", "subfield")
    mock_message2.text = "Required"
    
    mock_validator = Mock(spec=Schema)
    error = ValidationError(messages=[mock_message1, mock_message2])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid token with Field validator
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"
    
    # Test 2: Valid token with Schema validator
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(15, 0, 15))
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    validator = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test 4: ValidationError with type mismatch
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(10, 0, 10))
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(msg, 'start_position') for msg in messages)
        assert all(hasattr(msg, 'end_position') for msg in messages)
    
    # Test 5: Messages are sorted by start_position
    class MultiFieldSchema(Schema):
        field1 = String(allow_null=False)
        field2 = String(allow_null=False)
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(2, 0, 2))
    validator = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


class MockToken:
    def __init__(self, value, start=None, end=None):
        self.value = value
        self.start = start or Position(line=0, char_index=0)
        self.end = end or Position(line=0, char_index=len(str(value)))
    
    def lookup(self, index):
        return self


class MockValidator(Field):
    def __init__(self, should_fail=False, error_code="invalid", error_index=None):
        self.should_fail = should_fail
        self.error_code = error_code
        self.error_index = error_index or []
    
    def validate(self, value):
        if self.should_fail:
            message = Message(
                text="Invalid value",
                code=self.error_code,
                index=self.error_index
            )
            raise ValidationError(messages=[message])
        return value


def test_validate_with_positions_success():
    token = MockToken("test_value")
    validator = MockValidator(should_fail=False)
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"


def test_validate_with_positions_validation_error():
    token = MockToken("test_value")
    validator = MockValidator(should_fail=True, error_code="invalid", error_index=["field"])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    assert len(exc_info.value.messages()) > 0
    message = list(exc_info.value.messages())[0]
    assert message.code == "invalid"


def test_validate_with_positions_required_field():
    token = MockToken("test_value")
    validator = MockValidator(should_fail=True, error_code="required", error_index=["parent", "field"])
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = list(exc_info.value.messages())
    assert len(messages) > 0
    message = messages[0]
    assert message.code == "required"
    assert "field" in message.text
    assert "required" in message.text.lower()


def test_validate_with_positions_multiple_errors():
    token = MockToken("test_value")
    
    messages = [
        Message(text="Error 1", code="invalid", index=["field1"]),
        Message(text="Error 2", code="invalid", index=["field2"]),
    ]
    
    class MultiErrorValidator(Field):
        def validate(self, value):
            raise ValidationError(messages=messages)
    
    validator = MultiErrorValidator()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    error_messages = list(exc_info.value.messages())
    assert len(error_messages) == 2


def test_validate_with_positions_preserves_message_order():
    token = MockToken("test_value")
    
    messages = [
        Message(text="Error 2", code="invalid", index=["field2"]),
        Message(text="Error 1", code="invalid", index=["field1"]),
    ]
    
    class MultiErrorValidator(Field):
        def validate(self, value):
            raise ValidationError(messages=messages)
    
    validator = MultiErrorValidator()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    error_messages = list(exc_info.value.messages())
    assert len(error_messages) == 2


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest

    # Test 1: Successful validation without errors
    token = Mock(spec=Token)
    token.value = "valid_value"
    
    validator = Mock(spec=Field)
    validator.validate.return_value = "valid_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid_value"
    validator.validate.assert_called_once_with("valid_value")

    # Test 2: ValidationError with required field message
    token = Mock(spec=Token)
    token.value = {"name": ""}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup = Mock(return_value=token)
    
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.text = "This field is required."
    error_message.index = ["field", "name"]
    
    validator = Mock(spec=Field)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert "name" in messages[0].text
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end

    # Test 3: ValidationError with non-required field message
    token = Mock(spec=Token)
    token.value = {"age": "invalid"}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup = Mock(return_value=token)
    
    error_message = Mock(spec=Message)
    error_message.code = "invalid_type"
    error_message.text = "Expected an integer."
    error_message.index = ["age"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message])
    validation_error.messages = Mock(return_value=[error_message])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "invalid_type"
    assert messages[0].text == "Expected an integer."

    # Test 4: Multiple validation errors sorted by position
    token = Mock(spec=Token)
    token.value = {"field1": "", "field2": ""}
    
    token_at_5 = Mock(spec=Token)
    token_at_5.start = Mock(char_index=5)
    token_at_5.end = Mock(char_index=15)
    
    token_at_20 = Mock(spec=Token)
    token_at_20.start = Mock(char_index=20)
    token_at_20.end = Mock(char_index=30)
    
    token.lookup = Mock(side_effect=lambda idx: token_at_20 if idx == ["field2"] else token_at_5)
    
    error_message1 = Mock(spec=Message)
    error_message1.code = "required"
    error_message1.text = "Field required"
    error_message1.index = ["field2"]
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "required"
    error_message2.text = "Field required"
    error_message2.index = ["field1"]
    
    validator = Mock(spec=Schema)
    validation_error = ValidationError(messages=[error_message1, error_message2])
    validation_error.messages = Mock(return_value=[error_message1, error_message2])
    validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


# LLM-generated content at query #33
#--------------------------

def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test case 1: Successful validation with Field
    token = Token(value="test_value", start=Position(char_index=0, line_no=1, col_no=1), end=Position(char_index=10, line_no=1, col_no=11))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test case 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(char_index=0, line_no=1, col_no=1), end=Position(char_index=20, line_no=1, col_no=21))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test case 3: Validation error with required field
    class TestSchema2(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=Position(char_index=0, line_no=1, col_no=1), end=Position(char_index=10, line_no=1, col_no=11))
    schema = TestSchema2()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert any("required" in str(m) for m in messages)
    
    # Test case 4: Validation error with invalid type
    token = Token(value="not_an_int", start=Position(char_index=0, line_no=1, col_no=1), end=Position(char_index=10, line_no=1, col_no=11))
    field = Integer()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert all(hasattr(m, 'start_position') for m in messages)
    assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test case 5: Messages are sorted by start_position
    token = Token(value={"a": "invalid"}, start=Position(char_index=0, line_no=1, col_no=1), end=Position(char_index=20, line_no=1, col_no=21))
    
    class TestSchema3(Schema):
        a = Integer()
    
    schema = TestSchema3()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = exc_info.value.messages()
    positions = [m.start_position.char_index for m in messages]
    assert positions == sorted(positions)


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_validation_error_required_field():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["field1", "subfield"]
    mock_message.text = "Field is required"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'subfield' is required."
    assert messages[0].code == "required"
    mock_token.lookup.assert_called_once_with(["field1"])


def test_validate_with_positions_validation_error_other_code():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ["field1"]
    mock_message.text = "Invalid value"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    mock_token.lookup.assert_called_once_with(["field1"])


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by char_index
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_lookup_token1 = Mock(spec=Token)
    mock_lookup_token1.start = Mock(char_index=20)
    mock_lookup_token1.end = Mock(char_index=30)
    
    mock_lookup_token2 = Mock(spec=Token)
    mock_lookup_token2.start = Mock(char_index=5)
    mock_lookup_token2.end = Mock(char_index=15)
    
    def lookup_side_effect(index):
        if index == ["field1"]:
            return mock_lookup_token1
        elif index == ["field2"]:
            return mock_lookup_token2
        return mock_token
    
    mock_token.lookup.side_effect = lookup_side_effect
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "invalid"
    mock_message1.index = ["field1"]
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "invalid"
    mock_message2.index = ["field2"]
    mock_message2.text = "Error 2"
    
    mock_error = Mock(spec=ValidationError)
    mock_error.messages.return_value = [mock_message1, mock_message2]
    
    mock_validator = Mock(spec=Field)
    mock_validator.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


def test_validate_with_positions_with_schema():
    # Test with Schema validator
    mock_token = Mock(spec=Token)
    mock_token.value = {"key": "value"}
    
    mock_validator = Mock(spec=Schema)
    mock_validator.validate.return_value = {"key": "validated_value"}
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == {"key": "validated_value"}
    mock_validator.validate.assert_called_once_with({"key": "value"})


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Successful validation with Field
    token = Token(
        type="text",
        value="hello",
        start=Position(line=0, column=0, char_index=0),
        end=Position(line=0, column=5, char_index=5),
    )
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(
        type="text",
        value={"name": "John"},
        start=Position(line=0, column=0, char_index=0),
        end=Position(line=0, column=10, char_index=10),
    )
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(
        type="text",
        value={"name": None},
        start=Position(line=0, column=0, char_index=0),
        end=Position(line=0, column=10, char_index=10),
    )
    validator = RequiredSchema()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: ValidationError with invalid type
    token = Token(
        type="text",
        value="not_an_integer",
        start=Position(line=0, column=0, char_index=0),
        end=Position(line=0, column=14, char_index=14),
    )
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) > 0
        # Messages should be sorted by position
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index
    
    # Test 5: Multiple validation errors sorted by position
    class MultiFieldSchema(Schema):
        field1 = Integer()
        field2 = Integer()
    
    token = Token(
        type="text",
        value={"field1": "bad", "field2": "also_bad"},
        start=Position(line=0, column=0, char_index=0),
        end=Position(line=0, column=30, char_index=30),
    )
    validator = MultiFieldSchema()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = list(error.messages())
        assert len(messages) >= 1
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Valid token passes validation
    token = Token(value="test_value", start=0, end=10)
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test_value"
    
    # Test 2: Invalid token raises ValidationError with correct position
    token = Token(value=123, start=0, end=10)
    field = String()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        assert any(msg.start_position == 0 for msg in messages)
    
    # Test 3: Schema validation with required field error
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    schema = TestSchema()
    token = Token(value={"age": 25}, start=5, end=20)
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        assert any("required" in msg.code for msg in messages if msg.code == "required")
    
    # Test 4: Multiple validation errors are sorted by position
    class MultiFieldSchema(Schema):
        field1 = String()
        field2 = Integer()
        field3 = String()
    
    schema = MultiFieldSchema()
    token = Token(value={}, start=0, end=50)
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
        # Verify messages are sorted by position
        positions = [msg.start_position for msg in messages]
        assert positions == sorted(positions, key=lambda x: x.char_index if hasattr(x, 'char_index') else 0)
    
    # Test 5: Valid schema data passes validation
    class SimpleSchema(Schema):
        name = String()
    
    schema = SimpleSchema()
    token = Token(value={"name": "John"}, start=0, end=20)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_with_positions():
    import pytest
    from unittest.mock import Mock, MagicMock
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token

    # Test 1: Successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=String)
    mock_validator.validate.return_value = "test_value"

    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "test_value"
    mock_validator.validate.assert_called_once_with("test_value")

    # Test 2: ValidationError with required field
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_token

    mock_validator = Mock(spec=Schema)
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ("field1", "nested_field")
    error_message.text = "Required field"

    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error

    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)

    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert "nested_field" in messages[0].text
    assert messages[0].start_position == mock_token.start
    assert messages[0].end_position == mock_token.end

    # Test 3: ValidationError with non-required error
    mock_token = Mock(spec=Token)
    mock_token.value = "invalid"
    mock_token.start = Mock(char_index=5)
    mock_token.end = Mock(char_index=12)
    mock_token.lookup.return_value = mock_token

    mock_validator = Mock(spec=String)
    error_message = Mock(spec=Message)
    error_message.code = "invalid_type"
    error_message.index = ("field1",)
    error_message.text = "Invalid type"

    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error

    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)

    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "invalid_type"
    assert messages[0].text == "Invalid type"

    # Test 4: Multiple validation errors sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = {}
    mock_token.lookup.side_effect = lambda idx: Mock(
        spec=Token, start=Mock(char_index=idx[0] if idx else 0), end=Mock(char_index=10)
    )

    mock_validator = Mock(spec=Schema)
    error_msg1 = Mock(spec=Message)
    error_msg1.code = "invalid"
    error_msg1.index = (1,)
    error_msg1.text = "Error 1"

    error_msg2 = Mock(spec=Message)
    error_msg2.code = "required"
    error_msg2.index = (0, "field")
    error_msg2.text = "Error 2"

    validation_error = ValidationError(messages=[error_msg1, error_msg2])
    mock_validator.validate.side_effect = validation_error

    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)

    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_with_positions():
    from unittest.mock import Mock, MagicMock
    import pytest
    
    # Test 1: Successful validation without errors
    token = Mock(spec=Token)
    token.value = "valid_value"
    
    validator = Mock(spec=Field)
    validator.validate.return_value = "valid_value"
    
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid_value"
    validator.validate.assert_called_once_with("valid_value")
    
    # Test 2: Validation error with required field
    token = Mock(spec=Token)
    token.value = {}
    token.start = Mock(char_index=0)
    token.end = Mock(char_index=10)
    token.lookup.return_value = token
    
    message = Mock(spec=Message)
    message.code = "required"
    message.index = ["nested", "field"]
    message.text = "Field is required"
    
    error = ValidationError(messages=[message])
    validator = Mock(spec=Field)
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert "field" in messages[0].text.lower()
    assert messages[0].start_position == token.start
    assert messages[0].end_position == token.end
    
    # Test 3: Validation error with non-required field
    token = Mock(spec=Token)
    token.value = {"name": "invalid"}
    token.start = Mock(char_index=5)
    token.end = Mock(char_index=15)
    token.lookup.return_value = token
    
    message = Mock(spec=Message)
    message.code = "invalid"
    message.index = ["name"]
    message.text = "Invalid value"
    
    error = ValidationError(messages=[message])
    validator = Mock(spec=Schema)
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) > 0
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"
    
    # Test 4: Multiple validation errors sorted by position
    token = Mock(spec=Token)
    token.value = {}
    
    token1 = Mock(spec=Token)
    token1.start = Mock(char_index=20)
    token1.end = Mock(char_index=30)
    
    token2 = Mock(spec=Token)
    token2.start = Mock(char_index=5)
    token2.end = Mock(char_index=15)
    
    token.lookup.side_effect = [token2, token1]
    
    message1 = Mock(spec=Message)
    message1.code = "required"
    message1.index = ["field1"]
    message1.text = "Field1 required"
    
    message2 = Mock(spec=Message)
    message2.code = "invalid"
    message2.index = ["field2"]
    message2.text = "Field2 invalid"
    
    error = ValidationError(messages=[message1, message2])
    validator = Mock(spec=Schema)
    validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=validator)
    
    raised_error = exc_info.value
    messages = raised_error.messages()
    assert len(messages) == 2
    # Messages should be sorted by char_index
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


# LLM-generated content at query #39
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    import pytest

    # Test 1: Valid token with Field validator
    token = Token(value="hello", start=0, end=5)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 5})()
    
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"

    # Test 2: Valid token with integer
    token_int = Token(value="123", start=0, end=3)
    token_int.start = type('obj', (object,), {'char_index': 0})()
    token_int.end = type('obj', (object,), {'char_index': 3})()
    
    int_field = Integer()
    result = validate_with_positions(token=token_int, validator=int_field)
    assert result == 123

    # Test 3: Invalid token raises ValidationError with positions
    token_invalid = Token(value="not_a_number", start=0, end=12)
    token_invalid.start = type('obj', (object,), {'char_index': 0})()
    token_invalid.end = type('obj', (object,), {'char_index': 12})()
    
    int_field = Integer()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token_invalid, validator=int_field)
    
    error = exc_info.value
    messages = list(error.messages())
    assert len(messages) > 0
    assert all(hasattr(msg, 'start_position') for msg in messages)
    assert all(hasattr(msg, 'end_position') for msg in messages)

    # Test 4: Schema validation with required field error
    class TestSchema(Schema):
        name = String()
    
    token_schema = Token(value={}, start=0, end=2)
    token_schema.start = type('obj', (object,), {'char_index': 0})()
    token_schema.end = type('obj', (object,), {'char_index': 2})()
    token_schema.lookup = lambda x: token_schema
    
    schema = TestSchema()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token_schema, validator=schema)
    
    error = exc_info.value
    messages = list(error.messages())
    assert len(messages) >= 0
    
    # Test 5: Messages are sorted by start_position char_index
    token_multi = Token(value="invalid", start=0, end=7)
    token_multi.start = type('obj', (object,), {'char_index': 0})()
    token_multi.end = type('obj', (object,), {'char_index': 7})()
    
    int_field = Integer()
    try:
        validate_with_positions(token=token_multi, validator=int_field)
    except ValidationError as error:
        messages = list(error.messages())
        if len(messages) > 1:
            positions = [m.start_position.char_index for m in messages]
            assert positions == sorted(positions)


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Valid token with Field validator
    token = Token(value="test", start=0, end=4)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 4})()
    
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test 2: Valid token with Schema validator
    class TestSchema(Schema):
        name = String()
    
    token_value = {"name": "John"}
    token = Token(value=token_value, start=0, end=20)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 20})()
    
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: Invalid token - type validation error
    token = Token(value="not_an_int", start=0, end=10)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 10})()
    
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Required field error
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
    
    token = Token(value={}, start=0, end=5)
    token.start = type('obj', (object,), {'char_index': 0})()
    token.end = type('obj', (object,), {'char_index': 5})()
    token.lookup = lambda idx: token
    
    schema = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        # Verify messages are sorted by position
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #41
#--------------------------

```python
import pytest
from typesystem.base import Message, ValidationError
from typesystem.fields import Field, String, Integer
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token, Position


def test_validate_with_positions():
    # Test successful validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(0, 5, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"


def test_validate_with_positions_field_validation_error():
    # Test Field validation error
    token = Token(value="not_an_int", start=Position(0, 0, 0), end=Position(0, 10, 10))
    field = Integer()
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    assert len(exc_info.value.messages()) > 0


def test_validate_with_positions_required_field_error():
    # Test required field error
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={}, start=Position(0, 0, 0), end=Position(0, 2, 2))
    schema = TestSchema(required=True)
    
    with pytest.raises(ValidationError):
        validate_with_positions(token=token, validator=schema)


def test_validate_with_positions_schema_validation():
    # Test successful schema validation
    class TestSchema(Schema):
        name = String()
        age = Integer()
    
    token = Token(
        value={"name": "John", "age": 30},
        start=Position(0, 0, 0),
        end=Position(0, 30, 30)
    )
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John", "age": 30}


def test_validate_with_positions_error_sorting():
    # Test that validation errors are sorted by position
    class TestSchema(Schema):
        field1 = String()
        field2 = String()
    
    token = Token(
        value={"field1": 123, "field2": 456},
        start=Position(0, 0, 0),
        end=Position(0, 50, 50)
    )
    schema = TestSchema()
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=schema)
    
    messages = list(exc_info.value.messages())
    if len(messages) > 1:
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_message_attributes():
    # Test that positional message has correct attributes
    token = Token(value="test", start=Position(0, 0, 0), end=Position(0, 4, 4))
    field = String(max_length=2)
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = list(exc_info.value.messages())
    for message in messages:
        assert hasattr(message, 'text')
        assert hasattr(message, 'code')
        assert hasattr(message, 'start_position')
        assert hasattr(message, 'end_position')
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #42
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_result"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_result"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_with_required_error():
    # Test validation error with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "required"
    mock_message.index = ["parent", "field_name"]
    mock_message.text = "Field is required"
    
    mock_validator = Mock(spec=Field)
    error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "The field 'field_name' is required."
    assert messages[0].code == "required"
    mock_token.lookup.assert_called_once_with(["parent"])


def test_validate_with_positions_with_non_required_error():
    # Test validation error with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_message = Mock(spec=Message)
    mock_message.code = "invalid"
    mock_message.index = ["field"]
    mock_message.text = "Invalid value"
    
    mock_validator = Mock(spec=Schema)
    error = ValidationError(messages=[mock_message])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid value"
    assert messages[0].code == "invalid"
    mock_token.lookup.assert_called_once_with(["field"])


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple validation errors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_lookup_token1 = Mock(spec=Token)
    mock_lookup_token1.start = Mock(char_index=20)
    mock_lookup_token1.end = Mock(char_index=30)
    
    mock_lookup_token2 = Mock(spec=Token)
    mock_lookup_token2.start = Mock(char_index=5)
    mock_lookup_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_lookup_token2, mock_lookup_token1]
    
    mock_message1 = Mock(spec=Message)
    mock_message1.code = "invalid"
    mock_message1.index = ["field1"]
    mock_message1.text = "Error 1"
    
    mock_message2 = Mock(spec=Message)
    mock_message2.code = "invalid"
    mock_message2.index = ["field2"]
    mock_message2.text = "Error 2"
    
    mock_validator = Mock(spec=Schema)
    error = ValidationError(messages=[mock_message1, mock_message2])
    mock_validator.validate.side_effect = error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #43
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation with Field
    token = Token(value="hello", start=0, end=5)
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=0, end=15)
    validator = TestSchema()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"name": "John"}
    
    # Test 3: Validation error with Field
    token = Token(value="not_an_int", start=0, end=10)
    validator = Integer()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: Required field validation error
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={}, start=0, end=5)
    validator = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.code == "required" for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)
    
    # Test 5: Messages are sorted by start_position
    token = Token(value="invalid", start=0, end=7)
    validator = Integer()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        if len(messages) > 1:
            for i in range(len(messages) - 1):
                assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #44
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, Position
    
    # Test 1: Valid validation with Field
    token = Token(value="hello", start=Position(0, 0, 0), end=Position(5, 0, 5))
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"
    
    # Test 2: Valid validation with Schema
    class TestSchema(Schema):
        name = String()
    
    token = Token(value={"name": "John"}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    schema = TestSchema()
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}
    
    # Test 3: ValidationError with non-required code
    token = Token(value="not_a_number", start=Position(0, 0, 0), end=Position(12, 0, 12))
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(isinstance(m, Message) for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)
    
    # Test 4: ValidationError with required code
    class RequiredSchema(Schema):
        name = String(allow_null=False)
    
    token = Token(value={"name": None}, start=Position(0, 0, 0), end=Position(10, 0, 10))
    schema = RequiredSchema()
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.code == "required" for m in messages)
        assert any("required" in m.text.lower() or "field" in m.text.lower() for m in messages)
    
    # Test 5: Messages are sorted by char_index
    token = Token(value="invalid", start=Position(0, 0, 0), end=Position(7, 0, 7))
    field = Integer()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as error:
        messages = error.messages()
        positions = [m.start_position.char_index for m in messages if hasattr(m.start_position, 'char_index')]
        if len(positions) > 1:
            assert positions == sorted(positions)


# LLM-generated content at query #45
#--------------------------

```python
import pytest
from unittest.mock import Mock, MagicMock
from typesystem.base import Message, ValidationError
from typesystem.fields import Field
from typesystem.schemas import Schema
from typesystem.tokenize.tokens import Token


def test_validate_with_positions():
    # Test successful validation
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_validator = Mock(spec=Field)
    mock_validator.validate.return_value = "validated_value"
    
    result = validate_with_positions(token=mock_token, validator=mock_validator)
    assert result == "validated_value"
    mock_validator.validate.assert_called_once_with("test_value")


def test_validate_with_positions_required_field_error():
    # Test ValidationError with "required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=0)
    mock_lookup_token.end = Mock(char_index=10)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_validator = Mock(spec=Field)
    error_message = Mock(spec=Message)
    error_message.code = "required"
    error_message.index = ["field_name"]
    error_message.text = "Field is required"
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "required"
    assert "field_name" in messages[0].text
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end


def test_validate_with_positions_non_required_error():
    # Test ValidationError with non-"required" code
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    mock_token.start = Mock(char_index=0)
    mock_token.end = Mock(char_index=10)
    
    mock_lookup_token = Mock(spec=Token)
    mock_lookup_token.start = Mock(char_index=5)
    mock_lookup_token.end = Mock(char_index=15)
    mock_token.lookup.return_value = mock_lookup_token
    
    mock_validator = Mock(spec=Schema)
    error_message = Mock(spec=Message)
    error_message.code = "invalid"
    error_message.index = ["field"]
    error_message.text = "Invalid value"
    
    validation_error = ValidationError(messages=[error_message])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) > 0
    assert messages[0].code == "invalid"
    assert messages[0].text == "Invalid value"
    assert messages[0].start_position == mock_lookup_token.start
    assert messages[0].end_position == mock_lookup_token.end


def test_validate_with_positions_multiple_errors_sorted():
    # Test multiple ValidationErrors are sorted by position
    mock_token = Mock(spec=Token)
    mock_token.value = "test_value"
    
    mock_lookup_token1 = Mock(spec=Token)
    mock_lookup_token1.start = Mock(char_index=20)
    mock_lookup_token1.end = Mock(char_index=30)
    
    mock_lookup_token2 = Mock(spec=Token)
    mock_lookup_token2.start = Mock(char_index=5)
    mock_lookup_token2.end = Mock(char_index=15)
    
    mock_token.lookup.side_effect = [mock_lookup_token1, mock_lookup_token2]
    
    mock_validator = Mock(spec=Schema)
    error_message1 = Mock(spec=Message)
    error_message1.code = "invalid"
    error_message1.index = ["field1"]
    error_message1.text = "Error 1"
    
    error_message2 = Mock(spec=Message)
    error_message2.code = "invalid"
    error_message2.index = ["field2"]
    error_message2.text = "Error 2"
    
    validation_error = ValidationError(messages=[error_message1, error_message2])
    mock_validator.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=mock_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Messages should be sorted by start_position.char_index
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20


