####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_with_null_value_not_allowed():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value=None, start_index=0, end_index=3, content="null")
    validator = String()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_schema_required_field():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "name" in messages[0].text
        assert messages[0].start_position is not None


def test_validate_with_positions_with_schema_valid_data():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value="test", start_index=0, end_index=3, content="test")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={"name": "test"}, start_index=0, end_index=10, content='{"name":"test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}


def test_validate_with_positions_messages_sorted_by_char_index():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": String(), "field2": Integer()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions_with_required_field_error():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    class StringField(Field):
        errors = {"type": "Must be a string.", "required": "This field is required."}
        
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    schema = Schema(fields={"name": StringField(), "age": StringField()})
    token = MockToken(value={"name": "John"}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(m.code == "required" for m in messages)


def test_validate_with_positions_with_type_error():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": 123}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    class StringField(Field):
        errors = {"type": "Must be a string.", "required": "This field is required."}
        
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    schema = Schema(fields={"name": StringField()})
    token = MockToken(value={"name": 123}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(hasattr(m, 'start_position') for m in messages)
        assert all(hasattr(m, 'end_position') for m in messages)


def test_validate_with_positions_valid_input():
    from typesystem.base import ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": "John", "age": "30"}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    class StringField(Field):
        errors = {"type": "Must be a string.", "required": "This field is required."}
        
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    schema = Schema(fields={"name": StringField(), "age": StringField()})
    token = MockToken(value={"name": "John", "age": "30"}, start_index=0, end_index=20, content="test content here")
    
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John", "age": "30"}


def test_validate_with_positions_messages_sorted_by_char_index():
    from typesystem.base import ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    class StringField(Field):
        errors = {"type": "Must be a string.", "required": "This field is required."}
        
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value

    schema = Schema(fields={"field_a": StringField(), "field_b": StringField()})
    token = MockToken(value={}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        char_indices = [m.start_position.char_index for m in messages if m.start_position]
        assert char_indices == sorted(char_indices)


def test_validate_with_positions_preserves_message_attributes():
    from typesystem.base import ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return {"name": 123}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    class StringField(Field):
        errors = {"type": "Must be a string."}
        
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value

    schema = Schema(fields={"name": StringField()})
    token = MockToken(value={"name": 123}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert all(m.code == "type" for m in messages)
        assert all(m.index == ["name"] for m in messages)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    token = SimpleToken(value=42, start_index=0, end_index=1, content="42")
    validator = SimpleField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == 42


def test_validate_with_positions_with_validation_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=None, start_index=5, end_index=6, content="test")
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"type": "Invalid type"}
        
        def validate(self, value):
            raise self.validation_error("type")
    
    token = SimpleToken(value=None, start_index=0, end_index=4, content="value")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_required_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=None, start_index=5, end_index=10, content="content")
        
        def _get_key_token(self, key):
            return SimpleToken(value=None, start_index=0, end_index=4, content="field")
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    schema = Schema(fields={"name": SimpleField()})
    token = SimpleToken(value={}, start_index=0, end_index=2, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text.lower()
        assert messages[0].start_position is not None


def test_validate_with_positions_messages_sorted_by_char_index():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content="", child_index=None):
            super().__init__(value, start_index, end_index, content)
            self.child_index = child_index
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key == "field1":
                return SimpleToken(value=None, start_index=10, end_index=15, content=self._content)
            elif key == "field2":
                return SimpleToken(value=None, start_index=5, end_index=8, content=self._content)
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"type": "Invalid type"}
        
        def validate(self, value):
            raise self.validation_error("type")
    
    schema = Schema(fields={
        "field1": FailingField(),
        "field2": FailingField()
    })
    token = SimpleToken(value={"field1": "bad", "field2": "bad"}, start_index=0, end_index=20, content="content")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    # Create a mock Token class for testing
    class MockToken(Token):
        def _get_value(self):
            return {"name": ""}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    # Create a schema that requires a 'name' field
    schema = Schema(fields={"name": Field()})
    token = MockToken(value={"name": ""}, start_index=0, end_index=10, content="test content")
    
    # This should raise a ValidationError that gets caught at line 6
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        # If we reach here, the predicate at line 6 evaluated to True
        assert isinstance(error, ValidationError)
        assert len(error.messages()) > 0


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.base import Message
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self

    class SimpleField(Field):
        def validate(self, value):
            return value

    token = SimpleToken(value=42, start_index=0, end_index=1, content="42")
    field = SimpleField()
    result = validate_with_positions(token=token, validator=field)
    assert result == 42


def test_validate_with_positions_with_validation_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self

    class FailingField(Field):
        errors = {"custom": "Test error"}
        def validate(self, value):
            raise self.validation_error("custom")

    token = SimpleToken(value=42, start_index=0, end_index=1, content="42")
    field = FailingField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self

    class SimpleField(Field):
        def validate(self, value):
            return value

    schema = Schema(fields={"username": SimpleField()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text.lower()
        assert messages[0].start_position is not None


def test_validate_with_positions_nested_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content="", nested_token=None):
            super().__init__(value, start_index, end_index, content)
            self.nested_token = nested_token or self

        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self.nested_token
        def _get_key_token(self, key):
            return self.nested_token

    class FailingField(Field):
        errors = {"custom": "Nested error"}
        def validate(self, value):
            raise self.validation_error("custom")

    nested_token = SimpleToken(value="bad", start_index=5, end_index=7, content='{"x":"bad"}')
    token = SimpleToken(
        value={"x": "bad"},
        start_index=0,
        end_index=10,
        content='{"x":"bad"}',
        nested_token=nested_token
    )
    schema = Schema(fields={"x": FailingField()})
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_error_sorting_by_char_index():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content="", char_idx=0):
            super().__init__(value, start_index, end_index, content)
            self.char_idx = char_idx

        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self
        def _get_position(self, index):
            from typesystem.base import Position
            return Position(1, 1, self.char_idx)

    class FailingField(Field):
        errors = {"custom": "Error"}
        def validate(self, value):
            raise self.validation_error("custom")

    schema = Schema(fields={"a": FailingField(), "b": FailingField()})
    token = SimpleToken(
        value={"a": "bad1", "b": "bad2"},
        start_index=0,
        end_index=20,
        content='{"a":"bad1","b":"bad2"}',
        char_idx=0
    )
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = SimpleField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_with_validation_error():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class FailingField(Field):
        errors = {"custom": "Invalid value"}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_required_error():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key in self._value:
                return SimpleToken(
                    value=self._value[key],
                    start_index=self._start_index,
                    end_index=self._end_index,
                    content=self._content
                )
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": Field()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert any(msg.code == "required" for msg in messages)
        assert all(msg.start_position is not None for msg in messages)
        assert all(msg.end_position is not None for msg in messages)


def test_validate_with_positions_sorts_messages_by_char_index():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if isinstance(self._value, dict) and key in self._value:
                return SimpleToken(
                    value=self._value[key],
                    start_index=self._start_index + 5,
                    end_index=self._end_index + 5,
                    content=self._content
                )
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    token = SimpleToken(value={}, start_index=0, end_index=10, content="{field1:field2}")
    
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        char_indices = [msg.start_position.char_index for msg in messages]
        assert char_indices == sorted(char_indices)


def test_validate_with_positions_preserves_message_attributes():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class CustomField(Field):
        errors = {"custom": "Custom error"}
        
        def validate(self, value):
            error = ValidationError(
                messages=[Message(text="Custom error", code="custom", index=[])]
            )
            raise error
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = CustomField()
    
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert messages[0].code == "custom"
        assert messages[0].text == "Custom error"
        assert messages[0].index == []


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions_successful_validation():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Position
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
        
        errors = {"type": "Must be a string."}
    
    token = SimpleToken(value="hello", start_index=0, end_index=4, content="hello")
    field = StringField()
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"


def test_validate_with_positions_validation_error():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
        
        errors = {"type": "Must be a string."}
    
    token = SimpleToken(value=123, start_index=0, end_index=2, content="123")
    field = StringField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_nested_index():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError, Message
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(
                value=self._value.get(key) if isinstance(self._value, dict) else None,
                start_index=self._start_index,
                end_index=self._end_index,
                content=self._content
            )
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
        
        errors = {"type": "Must be a string."}
    
    token = SimpleToken(
        value={"name": 123},
        start_index=0,
        end_index=15,
        content='{"name": 123}'
    )
    field = StringField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0


def test_validate_with_positions_required_field_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(
                value=self._value.get(key) if isinstance(self._value, dict) else None,
                start_index=self._start_index,
                end_index=self._end_index,
                content=self._content
            )
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
        
        errors = {"type": "Must be a string."}
    
    schema = Schema(fields={"name": StringField()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)
        assert any("required" in msg.text.lower() for msg in messages)


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content, char_offset=0):
            super().__init__(value, start_index, end_index, content)
            self.char_offset = char_offset
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(
                value=self._value.get(key) if isinstance(self._value, dict) else None,
                start_index=self._start_index,
                end_index=self._end_index,
                content=self._content,
                char_offset=self.char_offset
            )
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
        
        errors = {"type": "Must be a string."}
    
    schema = Schema(fields={"name": StringField(), "email": StringField()})
    token = SimpleToken(value={"name": 123, "email": 456}, start_index=0, end_index=30, content='{"name": 123, "email": 456}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions_success():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_with_validation_error():
    from typesystem.fields import String, Integer
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="not_a_number", start_index=0, end_index=11, content="not_a_number")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=1, content="{}")
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text


def test_validate_with_positions_nested_error():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class NestedToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            if isinstance(key, str):
                return NestedToken(value="invalid", start_index=0, end_index=6, content="invalid")
            return self
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"age": Integer()})
    token = NestedToken(value={"age": "invalid"}, start_index=0, end_index=15, content='{"age": "invalid"}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class MultiFieldToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            if key == "field1":
                return MultiFieldToken(value=None, start_index=0, end_index=5, content="field1")
            elif key == "field2":
                return MultiFieldToken(value=None, start_index=10, end_index=15, content="field2")
            return self
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": String(), "field2": String()})
    token = MultiFieldToken(value={}, start_index=0, end_index=20, content="field1 field2")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import String
    from typesystem.base import Position
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value="test", start_index=0, end_index=3, content="test")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value="not_an_int", start_index=0, end_index=9, content="not_an_int")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return TestToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = TestToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text.lower()


def test_validate_with_positions_message_sorting():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return TestToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": String(), "field2": String()})
    token = TestToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_nested_schema_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def __init__(self, value, start_index, end_index, content="", nested_value=None):
            super().__init__(value, start_index, end_index, content)
            self.nested_value = nested_value
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if self.nested_value is not None:
                return TestToken(value=self.nested_value, start_index=0, end_index=5, content="nested")
            return TestToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    inner_schema = Schema(fields={"age": Integer()})
    outer_schema = Schema(fields={"person": inner_schema})
    token = TestToken(value={"person": {"age": "invalid"}}, start_index=0, end_index=10, content="{person:{}}", nested_value="invalid")
    
    try:
        validate_with_positions(token=token, validator=outer_schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return None
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test")
    
    token = MockToken(value=None, start_index=0, end_index=5, content="test")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        assert True


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test")

    class MockToken(Token):
        def __init__(self):
            super().__init__(value="test", start_index=0, end_index=4, content="test")

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return self

        def _get_key_token(self, key):
            return self

    token = MockToken()
    validator = FailingField()

    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError to be raised"
    except ValidationError as error:
        assert isinstance(error, ValidationError)
        assert len(error.messages()) > 0


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions_with_valid_token():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import String
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_with_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=3, content="test")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={}, start_index=0, end_index=3, content="test")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "name" in messages[0].text


def test_validate_with_positions_with_type_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Integer
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="not_an_int", start_index=0, end_index=9, content="not_an_int")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_error_messages_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key == "field1":
                return SimpleToken(value="invalid", start_index=0, end_index=6, content="invalid")
            return SimpleToken(value="test", start_index=10, end_index=13, content="test")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": Integer(), "field2": String()})
    token = SimpleToken(value={"field1": "invalid"}, start_index=0, end_index=20, content="test content here")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_preserves_message_index():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=3, content="test")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"username": String()})
    token = SimpleToken(value={}, start_index=0, end_index=3, content="test")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert messages[0].index == ["username"]


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleField(Field):
        def validate(self, value):
            if value is None:
                raise self.validation_error("null")
            return value
    
    class MockToken(Token):
        def __init__(self, value, content="test"):
            super().__init__(value, 0, len(content) - 1, content)
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    field = SimpleField()
    token = MockToken(None)
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError to be raised"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.base import Message
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = SimpleField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_validation_error_with_required():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            message = Message(text="This field is required.", code="required", index=["username"])
            raise ValidationError(messages=[message])
    
    token = SimpleToken(value={}, start_index=0, end_index=5, content="test{}")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert "username" in messages[0].text
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_validation_error_non_required():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            message = Message(text="Invalid value.", code="type", index=[])
            raise ValidationError(messages=[message])
    
    token = SimpleToken(value="invalid", start_index=0, end_index=6, content="invalid")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].text == "Invalid value."
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            msg1 = Message(text="Error 1.", code="error1", index=["field1"])
            msg2 = Message(text="Error 2.", code="error2", index=["field2"])
            raise ValidationError(messages=[msg2, msg1])
    
    token = SimpleToken(value={}, start_index=0, end_index=5, content="test{}")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index


def test_validate_with_positions_nested_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=self._value, start_index=self._start_index, end_index=self._end_index, content=self._content)
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            message = Message(text="Nested error.", code="nested", index=["parent", "child"])
            raise ValidationError(messages=[message])
    
    token = SimpleToken(value={"parent": {"child": "value"}}, start_index=0, end_index=10, content="nested_data")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].index == ["parent", "child"]
        assert messages[0].start_position is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    token = SimpleToken(value=42, start_index=0, end_index=2, content="42")
    validator = SimpleField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == 42


def test_validate_with_positions_with_validation_error():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"custom": "Custom error"}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    token = SimpleToken(value="invalid", start_index=0, end_index=7, content="invalid")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError, Message
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=None, start_index=0, end_index=0, content="")
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        errors = {"type": "Must be a string"}
        
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    
    schema = Schema(fields={"name": StringField()})
    token = SimpleToken(value={}, start_index=0, end_index=2, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"custom": "Error"}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken("test_value", 0, 10, "test_value")
    validator = SimpleField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"


def test_validate_with_positions_with_validation_error():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError, Message
    
    class FailingField(Field):
        errors = {"custom_error": "Custom error message"}
        
        def validate(self, value):
            raise self.validation_error("custom_error")
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken("invalid", 0, 7, "invalid")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_required_field_error():
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken({}, 0, 2, "{}")
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        errors = {"type": "Must be a string"}
        
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    
    schema = Schema(fields={"name": StringField()})
    token = SimpleToken({}, 0, 2, "{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError, Message
    
    class MultiErrorField(Field):
        errors = {"error1": "Error 1", "error2": "Error 2"}
        
        def validate(self, value):
            msg1 = Message(text="Error 1", code="error1", index=[])
            msg2 = Message(text="Error 2", code="error2", index=[])
            raise ValidationError(messages=[msg1, msg2])
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken("value", 0, 5, "value")
    validator = MultiErrorField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions_success():
    from typesystem.fields import String
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    
    class MockToken(Token):
        def _get_value(self):
            return "test_value"
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = MockToken(value="test_value", start_index=0, end_index=9, content="test_value")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test_value"


def test_validate_with_positions_validation_error():
    from typesystem.fields import Integer
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return "not_an_integer"
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = MockToken(value="not_an_integer", start_index=0, end_index=13, content="not_an_integer")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_schema():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}
        
        def _get_child_token(self, key):
            return MockToken(value="John", start_index=0, end_index=3, content="John")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = MockToken(value={"name": "John"}, start_index=0, end_index=20, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_required_field_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return {}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = MockToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)
        assert any("required" in msg.text for msg in messages)


def test_validate_with_positions_nested_schema():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content="", depth=0):
            super().__init__(value, start_index, end_index, content)
            self.depth = depth
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return MockToken(value="nested_value", start_index=0, end_index=11, content="nested_value", depth=self.depth + 1)
        
        def _get_key_token(self, key):
            return self
    
    inner_schema = Schema(fields={"inner_field": String()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = MockToken(value={"outer": {"inner_field": "value"}}, start_index=0, end_index=30, content='{"outer": {"inner_field": "value"}}')
    result = validate_with_positions(token=token, validator=outer_schema)
    assert result == {"outer": {"inner_field": "value"}}


def test_validate_with_positions_messages_sorted():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return {}
        
        def _get_child_token(self, key):
            return MockToken(value="", start_index=0, end_index=0, content="")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": String(), "field2": String()})
    token = MockToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        char_indices = [msg.start_position.char_index for msg in messages]
        assert char_indices == sorted(char_indices)


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.base import Message, ValidationError, Position
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Field
    
    class MockToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test_code")
    
    token = MockToken(value="test", start_index=0, end_index=3, content="test")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError to be raised"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions_with_valid_token():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value=42, start_index=0, end_index=5, content="hello")
    field = SimpleField()
    result = validate_with_positions(token=token, validator=field)
    assert result == 42


def test_validate_with_positions_with_validation_error():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class FailingField(Field):
        errors = {"custom": "Invalid value"}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value=42, start_index=0, end_index=5, content="hello")
    field = FailingField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_required_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": SimpleField()})
    token = SimpleToken(value={}, start_index=0, end_index=5, content="hello")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": SimpleField(), "field2": SimpleField()})
    token = SimpleToken(value={}, start_index=0, end_index=5, content="hello")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions_success():
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class SimpleField(Field):
        def validate(self, value):
            return value
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    field = SimpleField()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"


def test_validate_with_positions_validation_error():
    from typesystem.fields import Field
    from typesystem.base import ValidationError, Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"custom": "Invalid value."}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    field = FailingField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].text == "Invalid value."


def test_validate_with_positions_required_field_error():
    from typesystem.fields import Field
    from typesystem.base import ValidationError, Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=0, content="")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": Field()})
    token = SimpleToken(value={}, start_index=0, end_index=0, content="")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert any("required" in msg.code for msg in messages)


def test_validate_with_positions_with_nested_index():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    
    class NestedToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return NestedToken(value={}, start_index=5, end_index=10, content="nested")
        
        def _get_key_token(self, key):
            return self
    
    class FailingField(Field):
        errors = {"custom": "Invalid value."}
        
        def validate(self, value):
            raise self.validation_error("custom")
    
    token = NestedToken(value={"nested": "value"}, start_index=0, end_index=15, content="nested_content")
    field = FailingField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_message_sorting():
    from typesystem.fields import Field
    from typesystem.base import ValidationError, Message, Position
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    
    class SortingToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SortingToken(value={}, start_index=0, end_index=5, content="token")
        
        def _get_key_token(self, key):
            return self
    
    class MultiErrorField(Field):
        def validate(self, value):
            messages = [
                Message(text="Error 1", code="custom", index=["field1"]),
                Message(text="Error 2", code="custom", index=["field2"]),
            ]
            raise ValidationError(messages=messages)
    
    token = SortingToken(value={}, start_index=0, end_index=10, content="test_content")
    field = MultiErrorField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert all(msg.start_position is not None for msg in messages)


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.fields import Field
    from typesystem.base import ValidationError, Message
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test")
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test_value", start_index=0, end_index=9, content="test_value")
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return "test_value"
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = MockToken(value="test_value", start_index=0, end_index=9, content="test_value")
    
    class FailingField:
        def validate(self, value):
            raise ValidationError(text="Test error", code="test")
    
    field = FailingField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    validator = String()
    token = Token(value="hello", start_index=0, end_index=4, content="hello")
    result = validate_with_positions(token=token, validator=validator)
    assert result == "hello"


def test_validate_with_positions_with_schema():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    schema = Schema(fields={"name": String()})
    token = Token(value={"name": "John"}, start_index=0, end_index=16, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_invalid_type():
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    validator = Integer()
    token = Token(value="not_an_int", start_index=0, end_index=9, content="not_an_int")
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"name": String()})
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "name" in messages[0].index
        assert messages[0].start_position is not None


def test_validate_with_positions_nested_schema():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    inner_schema = Schema(fields={"first": String()})
    outer_schema = Schema(fields={"user": inner_schema})
    token = Token(
        value={"user": {"first": "John"}},
        start_index=0,
        end_index=25,
        content='{"user": {"first": "John"}}'
    )
    result = validate_with_positions(token=token, validator=outer_schema)
    assert result == {"user": {"first": "John"}}


def test_validate_with_positions_message_sorting():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    token = Token(
        value={"name": 123, "age": "not_an_int"},
        start_index=0,
        end_index=30,
        content='{"name": 123, "age": "not_an_int"}'
    )
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        errors = {"type": "Must be a string."}
        
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    field = StringField()
    
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"


def test_validate_with_positions_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class StringField(Field):
        errors = {"type": "Must be a string."}
        
        def validate(self, value):
            if isinstance(value, str):
                return value
            raise self.validation_error("type")
    
    token = SimpleToken(value=123, start_index=0, end_index=2, content="123")
    field = StringField()
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field, String
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    class DummyField(Field):
        def validate(self, value):
            return value
    
    schema = Schema(fields={"name": DummyField()})
    token = SimpleToken(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert any(msg.code == "required" for msg in messages)


def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            nested_value = self._value.get(key, {}) if isinstance(self._value, dict) else {}
            return SimpleToken(value=nested_value, start_index=0, end_index=5, content="nested")
        
        def _get_key_token(self, key):
            return self
    
    class DummyField(Field):
        def validate(self, value):
            return value
    
    schema = Schema(fields={"user": Schema(fields={"name": DummyField()})})
    token = SimpleToken(value={"user": {}}, start_index=0, end_index=10, content="user object")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert all(msg.start_position is not None for msg in messages)


def test_validate_with_positions_message_sorting():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value={}, start_index=0, end_index=1, content="{}")
        
        def _get_key_token(self, key):
            return self
    
    class DummyField(Field):
        def validate(self, value):
            return value
    
    schema = Schema(fields={"field1": DummyField(), "field2": DummyField()})
    token = SimpleToken(value={}, start_index=0, end_index=2, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        char_indices = [msg.start_position.char_index for msg in messages]
        assert char_indices == sorted(char_indices)


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return {"name": ""}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = MockToken(value={"name": ""}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions_successful_validation():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Field, String
    from typesystem.schemas import Schema
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value="test", start_index=0, end_index=3, content="test")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"


def test_validate_with_positions_with_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Integer
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value="not_an_integer", start_index=0, end_index=13, content="not_an_integer")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) > 0
        assert error.messages()[0].start_position is not None
        assert error.messages()[0].end_position is not None


def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value={}, start_index=0, end_index=1, content="{}")
    validator = Schema(fields={"name": String()})
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "required" in messages[0].text


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return TestToken(value=self._value.get(key), start_index=0, end_index=5, content="field")
        
        def _get_key_token(self, key):
            return self
    
    token = TestToken(value={"a": "text", "b": "text"}, start_index=0, end_index=20, content="{'a': 'text', 'b': 'text'}")
    validator = Schema(fields={"a": Integer(), "b": Integer()})
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_with_nested_schema():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import String
    from typesystem.base import ValidationError
    
    class TestToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if isinstance(self._value, dict) and key in self._value:
                return TestToken(value=self._value[key], start_index=0, end_index=5, content="nested")
            return self
        
        def _get_key_token(self, key):
            return self
    
    inner_schema = Schema(fields={"name": String()})
    outer_schema = Schema(fields={"user": inner_schema})
    token = TestToken(value={"user": {}}, start_index=0, end_index=10, content="{'user': {}}")
    
    try:
        validate_with_positions(token=token, validator=outer_schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions_raises_validation_error():
    from typesystem.base import ValidationError, Message
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test_code")
    
    class MockToken(Token):
        def __init__(self):
            super().__init__(value="test", start_index=0, end_index=4, content="test_content")
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = MockToken()
    validator = FailingField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={"name": "John"}, start_index=0, end_index=20, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_with_required_field_missing():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key == "name":
                return SimpleToken(value=None, start_index=0, end_index=5, content='{}')
            return self
        
        def _get_key_token(self, key):
            return self

    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={}, start_index=0, end_index=2, content='{}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert messages[0].index == ["name"]
        assert "required" in messages[0].text.lower()


def test_validate_with_positions_with_type_error():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key == "age":
                return SimpleToken(value="not_an_int", start_index=7, end_index=20, content='{"age": "not_an_int"}')
            return self
        
        def _get_key_token(self, key):
            return self

    schema = Schema(fields={"age": Integer()})
    token = SimpleToken(value={"age": "not_an_int"}, start_index=0, end_index=21, content='{"age": "not_an_int"}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            if key == "field1":
                return SimpleToken(value=None, start_index=0, end_index=10, content='{"field1": null, "field2": null}')
            elif key == "field2":
                return SimpleToken(value=None, start_index=15, end_index=25, content='{"field1": null, "field2": null}')
            return self
        
        def _get_key_token(self, key):
            return self

    schema = Schema(fields={"field1": String(), "field2": String()})
    token = SimpleToken(value={}, start_index=0, end_index=32, content='{"field1": null, "field2": null}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_with_field_validator():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    field = String(max_length=5)
    token = SimpleToken(value="Hello", start_index=0, end_index=6, content='"Hello"')
    result = validate_with_positions(token=token, validator=field)
    assert result == "Hello"


def test_validate_with_positions_with_field_validator_error():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError

    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self

    field = String(max_length=5)
    token = SimpleToken(value="Hello World", start_index=0, end_index=12, content='"Hello World"')
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    schema = Schema(fields={"name": String()})
    token = Token(value={"name": "John"}, start_index=0, end_index=16, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}


def test_validate_with_positions_null_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"name": String()})
    token = Token(value=None, start_index=0, end_index=4, content="null")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "null"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_required_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"name": String()})
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert "required" in messages[0].text.lower()
        assert messages[0].start_position is not None


def test_validate_with_positions_type_error():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"age": Integer()})
    token = Token(value={"age": "not_a_number"}, start_index=0, end_index=20, content='{"age": "not_a_number"}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_field_validator():
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    field = String()
    token = Token(value="hello", start_index=0, end_index=6, content='"hello"')
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"


def test_validate_with_positions_multiple_errors():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import ValidationError
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    token = Token(value={"age": "invalid"}, start_index=0, end_index=20, content='{"age": "invalid"}')
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1
        for message in messages:
            assert message.start_position is not None
            assert message.end_position is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_valid_value():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken(value="test", start_index=0, end_index=3, content="test")
    field = String()
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"


def test_validate_with_positions_with_schema_error():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.error import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=None, start_index=0, end_index=10, content=self._content)
        
        def _get_key_token(self, key):
            return SimpleToken(value=None, start_index=0, end_index=5, content=self._content)
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken(value={}, start_index=0, end_index=10, content="test content")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_non_required_error():
    from typesystem.fields import Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.error import ValidationError
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken(value=self._value.get(key), start_index=5, end_index=10, content=self._content)
        
        def _get_key_token(self, key):
            return SimpleToken(value=None, start_index=0, end_index=5, content=self._content)
    
    schema = Schema(fields={"age": Integer()})
    token = SimpleToken(value={"age": "invalid"}, start_index=0, end_index=20, content="test content here")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_messages_sorted_by_char_index():
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.error import ValidationError
    
    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content="", char_offset=0):
            super().__init__(value, start_index, end_index, content)
            self.char_offset = char_offset
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            offset = 10 if key == "field2" else 0
            return SimpleToken(value=self._value.get(key), start_index=offset, end_index=offset+5, content=self._content, char_offset=offset)
        
        def _get_key_token(self, key):
            offset = 10 if key == "field2" else 0
            return SimpleToken(value=None, start_index=offset, end_index=offset+5, content=self._content, char_offset=offset)
    
    schema = Schema(fields={"field1": String(), "field2": String()})
    token = SimpleToken(value={}, start_index=0, end_index=20, content="test content here now")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions_with_valid_value():
    from typesystem.base import Message
    from typesystem.fields import String
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken("valid_string", 0, 11, "valid_string")
    validator = String()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid_string"


def test_validate_with_positions_with_type_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import Integer
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    token = SimpleToken("not_an_int", 0, 9, "not_an_int")
    validator = Integer()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "type"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


def test_validate_with_positions_with_required_error():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken({}, 0, 1, "{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken({}, 0, 1, "{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) > 0
        assert messages[0].code == "required"
        assert "name" in messages[0].text


def test_validate_with_positions_messages_sorted_by_char_index():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def __init__(self, value, start_index, end_index, content="", child_tokens=None):
            super().__init__(value, start_index, end_index, content)
            self.child_tokens = child_tokens or {}
        
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return self.child_tokens.get(key, SimpleToken({}, 0, 1, "{}"))
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"field1": String(), "field2": Integer()})
    token = SimpleToken({}, 0, 10, "{field1,field2}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 2
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


def test_validate_with_positions_preserves_message_properties():
    from typesystem.base import Message, ValidationError
    from typesystem.fields import String
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    
    class SimpleToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return SimpleToken({}, 0, 1, "{}")
        
        def _get_key_token(self, key):
            return self
    
    schema = Schema(fields={"name": String()})
    token = SimpleToken({}, 0, 1, "{}")
    
    try:
        validate_with_positions(token=token, validator=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert messages[0].code == "required"
        assert messages[0].index == ["name"]
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions_catches_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    class MockToken(Token):
        def _get_value(self):
            return {"name": ""}
        
        def _get_child_token(self, key):
            return self
        
        def _get_key_token(self, key):
            return self
    
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Test error", code="test")
    
    token = MockToken(value={"name": ""}, start_index=0, end_index=10, content="test content")
    validator = MockField()
    
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Expected ValidationError to be raised"
    except ValidationError as error:
        assert error is not None
        assert isinstance(error, ValidationError)


