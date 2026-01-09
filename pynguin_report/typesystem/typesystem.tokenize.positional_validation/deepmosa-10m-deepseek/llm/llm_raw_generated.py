####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={}, start_index=0, end_index=0, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "The field 'name' is required."
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 1
    assert message.end_position.char_index == 0

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value="not an object", start_index=0, end_index=12, content='"not an object"')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Must be an object."
    assert message.code == "type"
    assert message.index == []
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 13
    assert message.end_position.char_index == 12

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    nested_schema = Schema(fields={"inner": MockField()})
    schema = Schema(fields={"outer": nested_schema})
    token = Token(value={"outer": {}}, start_index=0, end_index=15, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]
    outer_token = token.lookup(["outer"])
    assert message.start_position == outer_token.start
    assert message.end_position == outer_token.end

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    schema = Schema(fields={"name": MockField()})
    token = Token(value={"name": "bad"}, start_index=0, end_index=15, content='{"name": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Invalid value."
    assert message.code == "invalid"
    assert message.index == ["name"]
    name_token = token.lookup(["name"])
    assert message.start_position == name_token.start
    assert message.end_position == name_token.end

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    schema = Schema(fields={"a": MockField(), "b": MockField()})
    token = Token(value={"b": "bad", "a": "bad"}, start_index=0, end_index=23, content='{"b": "bad", "a": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 2
    first_message = messages[0]
    second_message = messages[1]
    assert first_message.index == ["b"]
    assert second_message.index == ["a"]
    assert first_message.start_position.char_index < second_message.start_position.char_index

def test_validate_with_positions_allow_null():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": MockField()}, allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={123: "value"}, start_index=0, end_index=15, content='{123: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, type(e))
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "All object keys must be strings."
    assert message.code == "invalid_key"
    assert message.index == [123]
    key_token = token.lookup_key([123])
    assert message.start_position == key_token.start
    assert message.end_position == key_token.end


# LLM-generated content at query #2
#--------------------------

def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typing

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._child_tokens = {}
            self._key_tokens = {}

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return self._child_tokens[key]

        def _get_key_token(self, key):
            return self._key_tokens[key]

        def set_child_token(self, key, token):
            self._child_tokens[key] = token

        def set_key_token(self, key, token):
            self._key_tokens[key] = token

    class MockField(Field):
        errors = {"required": "This field is required."}

        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required")

    field = MockField()
    schema = Schema(fields={"field1": field})
    token = MockToken(value={"field1": None})
    child_token = MockToken(value=None, start_index=10, end_index=20, content=" " * 30)
    token.set_child_token(["field1"], child_token)
    key_token = MockToken(value="field1", start_index=5, end_index=10, content=" " * 30)
    token.set_key_token("field1", key_token)
    try:
        from typesystem.tokenize.positional_validation import validate_with_positions
        result = validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field1"]
        assert message.start_position.char_index == 5
        assert message.end_position.char_index == 10
        assert message.text == "The field 'field1' is required."


# LLM-generated content at query #3
#--------------------------

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    token = Token(value="test", start_index=0, end_index=3, content="test")
    validator = MockField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "test"

def test_validate_with_positions_validation_error_without_index():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Invalid value", code="invalid")])
    token = Token(value="bad", start_index=0, end_index=2, content="bad")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "Invalid value"
        assert msg.code == "invalid"
        assert msg.index == []
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_validation_error_with_index():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Field error", code="error", index=["key"])])
    class MockToken(Token):
        def _get_child_token(self, key):
            return Token(value="nested", start_index=5, end_index=9, content="content")
    token = MockToken(value={"key": "val"}, start_index=0, end_index=10, content='{"key": "val"}')
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "Field error"
        assert msg.code == "error"
        assert msg.index == ["key"]
        assert msg.start_position.char_index == 5
        assert msg.end_position.char_index == 9

def test_validate_with_positions_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(messages=[Message(text="This field is required.", code="required", index=["field"])])
    class MockToken(Token):
        def _get_child_token(self, key):
            return Token(value={}, start_index=1, end_index=2, content="{}")
    token = MockToken(value={}, start_index=0, end_index=2, content="{}")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "The field 'field' is required."
        assert msg.code == "required"
        assert msg.index == ["field"]
        assert msg.start_position.char_index == 1
        assert msg.end_position.char_index == 2

def test_validate_with_positions_multiple_messages_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(messages=[
                Message(text="Error 2", code="error2", index=["b"]),
                Message(text="Error 1", code="error1", index=["a"])
            ])
    class MockToken(Token):
        def _get_child_token(self, key):
            if key == "a":
                return Token(value=1, start_index=10, end_index=14, content="content")
            if key == "b":
                return Token(value=2, start_index=5, end_index=9, content="content")
            return Token(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value={"a":1,"b":2}, start_index=0, end_index=20, content='{"a":1,"b":2}')
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].code == "error1"
        assert messages[0].start_position.char_index == 10
        assert messages[1].code == "error2"
        assert messages[1].start_position.char_index == 5


# LLM-generated content at query #4
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
    token = MockToken(value={}, start_index=0, end_index=10, content='')
    field = Field()
    schema = Schema(fields={'field': field})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'field' is required."
        assert msg.code == 'required'
        assert msg.index == ['field']
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {'outer': {}}
        def _get_child_token(self, key):
            if key == 'outer':
                return MockToken(value={}, start_index=7, end_index=12, content='')
            return MockToken(value=None, start_index=0, end_index=5, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
    token = MockToken(value={'outer': {}}, start_index=0, end_index=20, content='')
    inner_field = Field()
    inner_schema = Schema(fields={'inner': inner_field})
    outer_schema = Schema(fields={'outer': inner_schema})
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'inner' is required."
        assert msg.code == 'required'
        assert msg.index == ['outer', 'inner']
        assert msg.start_position.char_index == 7
        assert msg.end_position.char_index == 12

def test_validate_with_positions_non_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {'field': 'invalid'}
        def _get_child_token(self, key):
            return MockToken(value='invalid', start_index=8, end_index=15, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
    token = MockToken(value={'field': 'invalid'}, start_index=0, end_index=20, content='')
    class CustomField(Field):
        errors = {'custom': 'Custom error'}
        def validate(self, value):
            raise self.validation_error('custom')
    field = CustomField()
    schema = Schema(fields={'field': field})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == 'Custom error'
        assert msg.code == 'custom'
        assert msg.index == ['field']
        assert msg.start_position.char_index == 8
        assert msg.end_position.char_index == 15

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=''):
            super().__init__(value, start_index, end_index, content)
            self._child_tokens = {}
        def _get_value(self):
            return {'a': 'invalid', 'b': 'invalid'}
        def _get_child_token(self, key):
            if key == 'a':
                return MockToken(value='invalid', start_index=4, end_index=10, content='')
            elif key == 'b':
                return MockToken(value='invalid', start_index=14, end_index=20, content='')
            return MockToken(value=None, start_index=0, end_index=5, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
    token = MockToken(value={'a': 'invalid', 'b': 'invalid'}, start_index=0, end_index=25, content='')
    class CustomField(Field):
        errors = {'custom': 'Custom error'}
        def validate(self, value):
            raise self.validation_error('custom')
    field_a = CustomField()
    field_b = CustomField()
    schema = Schema(fields={'a': field_a, 'b': field_b})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 4
        assert messages[1].start_position.char_index == 14

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return 'invalid'
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content='')
    token = MockToken(value='invalid', start_index=0, end_index=7, content='')
    class StringField(Field):
        errors = {'type': 'Must be a string.'}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error('type')
            return value
    class IntField(Field):
        errors = {'type': 'Must be an integer.'}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error('type')
            return value
    union = Union(any_of=[StringField(), IntField()])
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == 'Did not match any valid type.'
        assert msg.code == 'union'
        assert msg.index == []
        assert msg.start_position == token.start
        assert msg.end_position == token.end


# LLM-generated content at query #5
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'name' is required."
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 12
    assert message.end_position.char_index == 11

def test_validate_with_positions_schema_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={123: "value"}, start_index=0, end_index=12, content='{123: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "All object keys must be strings."
    assert message.code == "invalid_key"
    assert message.index == [123]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 14
    assert message.end_position.char_index == 13

def test_validate_with_positions_field_type_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    field = MockField()
    token = Token(value=123, start_index=0, end_index=2, content="123")
    try:
        validate_with_positions(token=token, validator=field)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be a string."
    assert message.code == "type"
    assert message.index == []
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 3
    assert message.end_position.char_index == 2

def test_validate_with_positions_schema_nested_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_schema = Schema(fields={"inner": MockField()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": 123}}, start_index=0, end_index=24, content='{"outer": {"inner": 123}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be a string."
    assert message.code == "type"
    assert message.index == ["outer", "inner"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 25
    assert message.end_position.char_index == 24

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field, Union
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntegerField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[StringField(), IntegerField()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Did not match any valid type."
    assert message.code == "union"
    assert message.index == []
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 4
    assert message.end_position.char_index == 3

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"null": "May not be null."}
        def validate(self, value):
            if value is None and not self.allow_null:
                raise self.validation_error("null")
            return value
    field = MockField(allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=field)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"null": "May not be null."}
        def validate(self, value):
            if value is None and not self.allow_null:
                raise self.validation_error("null")
            return value
    field = MockField(allow_null=False)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    try:
        validate_with_positions(token=token, validator=field)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    import typesystem

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken(self._value.get(key), 0, 0)
        def _get_key_token(self, key):
            return MockToken(key, 0, 0)

    field = MockField()
    schema = Schema(fields={"field": field})
    token = MockToken({"field": None})
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.exceptions.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.text == "The field 'field' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #7
#--------------------------

def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken(self._value.get(key), 0, 0)
        def _get_key_token(self, key):
            return MockToken(key, 0, 0)
    class MockField(Field):
        def validate(self, value):
            raise self.validation_error("required")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = MockToken({}, 0, 0)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.text == "The field 'field' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #8
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"existing": "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="")
    token = MockToken(value={"existing": "value"}, start_index=0, end_index=20, content="")
    fields = {"required_field": Field(allow_null=False)}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["required_field"]
        assert messages[0].text == "The field 'required_field' is required."
        assert messages[0].start_position == token.lookup([]).start
        assert messages[0].end_position == token.lookup([]).end

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "not an object"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="not an object", start_index=0, end_index=15, content="")
    fields = {"some_field": Field(allow_null=False)}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"
        assert messages[0].index == []
        assert messages[0].text == "Must be an object."
        assert messages[0].start_position == token.start
        assert messages[0].end_position == token.end

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"nested": {"field": "invalid"}}
        def _get_child_token(self, key):
            if key == "nested":
                return MockNestedToken(value={"field": "invalid"}, start_index=10, end_index=30, content="")
            elif key == "field":
                return MockNestedToken(value="invalid", start_index=20, end_index=25, content="")
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    class MockNestedToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            if key == "field":
                return MockNestedToken(value="invalid", start_index=20, end_index=25, content="")
            return MockNestedToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockNestedToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value={"nested": {"field": "invalid"}}, start_index=0, end_index=40, content="")
    nested_field = Field(allow_null=False)
    nested_schema = Schema(fields={"field": nested_field})
    schema = Schema(fields={"nested": nested_schema})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].index == ["nested", "field"]
        assert messages[0].start_position == token.lookup(["nested", "field"]).start
        assert messages[0].end_position == token.lookup(["nested", "field"]).end

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
            self._mock_child_tokens = {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self._mock_child_tokens.get(key, MockToken(value=None, start_index=0, end_index=0, content=""))
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def set_child_token(self, key, token):
            self._mock_child_tokens[key] = token
    token = MockToken(value={"a": 1, "b": 2}, start_index=0, end_index=20, content="")
    child_a = MockToken(value=1, start_index=5, end_index=6, content="")
    child_b = MockToken(value=2, start_index=12, end_index=13, content="")
    token.set_child_token("a", child_a)
    token.set_child_token("b", child_b)
    fields = {"a": Field(allow_null=False), "b": Field(allow_null=False)}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 0

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="invalid", start_index=0, end_index=7, content="")
    union = Union(any_of=[Field(allow_null=False), Field(allow_null=False)])
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "union"
        assert messages[0].index == []
        assert messages[0].text == "Did not match any valid type."
        assert messages[0].start_position == token.start
        assert messages[0].end_position == token.end


# LLM-generated content at query #9
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    msg = error.messages()[0]
    assert msg.text == "The field 'name' is required."
    assert msg.code == "required"
    assert msg.index == ["name"]
    assert msg.start_position.line_no == 1
    assert msg.start_position.column_no == 1
    assert msg.start_position.char_index == 0
    assert msg.end_position.line_no == 1
    assert msg.end_position.column_no == 12
    assert msg.end_position.char_index == 11

def test_validate_with_positions_schema_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        pass
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={1: "value"}, start_index=0, end_index=10, content='{1: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    msg = error.messages()[0]
    assert msg.text == "All object keys must be strings."
    assert msg.code == "invalid_key"
    assert msg.index == [1]
    assert msg.start_position.line_no == 1
    assert msg.start_position.column_no == 1
    assert msg.start_position.char_index == 0
    assert msg.end_position.line_no == 1
    assert msg.end_position.column_no == 12
    assert msg.end_position.char_index == 11

def test_validate_with_positions_field_type_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    field = MockField()
    token = Token(value=123, start_index=0, end_index=2, content="123")
    try:
        validate_with_positions(token=token, validator=field)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    msg = error.messages()[0]
    assert msg.text == "Must be a string."
    assert msg.code == "type"
    assert msg.index == []
    assert msg.start_position.line_no == 1
    assert msg.start_position.column_no == 1
    assert msg.start_position.char_index == 0
    assert msg.end_position.line_no == 1
    assert msg.end_position.column_no == 3
    assert msg.end_position.char_index == 2

def test_validate_with_positions_schema_nested_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_schema = Schema(fields={"inner": MockField()})
    outer_schema = Schema(fields={"nested": inner_schema})
    token = Token(value={"nested": {"inner": 456}}, start_index=0, end_index=25, content='{"nested": {"inner": 456}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    msg = error.messages()[0]
    assert msg.text == "Must be a string."
    assert msg.code == "type"
    assert msg.index == ["nested", "inner"]
    assert msg.start_position.line_no == 1
    assert msg.start_position.column_no == 13
    assert msg.start_position.char_index == 12
    assert msg.end_position.line_no == 1
    assert msg.end_position.column_no == 25
    assert msg.end_position.char_index == 24

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[StringField(), IntField()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    msg = error.messages()[0]
    assert msg.text == "Did not match any valid type."
    assert msg.code == "union"
    assert msg.index == []
    assert msg.start_position.line_no == 1
    assert msg.start_position.column_no == 1
    assert msg.start_position.char_index == 0
    assert msg.end_position.line_no == 1
    assert msg.end_position.column_no == 4
    assert msg.end_position.char_index == 3

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}


# LLM-generated content at query #10
#--------------------------

def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken(self._value.get(key), 0, 0)
        def _get_key_token(self, key):
            return MockToken(key, 0, 0)
        def lookup(self, index):
            token = self
            for key in index:
                token = token._get_child_token(key)
            return token
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = MockToken({"field": None})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.text == "The field 'field' is required."
        assert message.start_position is not None
        assert message.end_position is not None
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #11
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.text == "The field 'name' is required."

def test_validate_with_positions_schema_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={123: "value"}, start_index=0, end_index=12, content='{123: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid_key"
    assert message.index == [123]

def test_validate_with_positions_schema_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "bad"}, start_index=0, end_index=15, content='{"name": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid"
    assert message.index == ["name"]
    assert message.text == "Invalid value."

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field, Union
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class MockField2(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[MockField1(), MockField2()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "union"

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value.upper()
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "TEST"}

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid."}
        def validate(self, value):
            raise self.validation_error("invalid")
    inner_schema = Schema(fields={"inner": MockField()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": "bad"}}, start_index=0, end_index=30, content='{"outer": {"inner": "bad"}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid"
    assert message.index == ["outer", "inner"]
    assert message.text == "Invalid."

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid."}
        def validate(self, value):
            raise self.validation_error("invalid")
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=15, content='{"a": 1, "b": 2}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].index == ["a"]
    assert messages[1].index == ["b"]


# LLM-generated content at query #12
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={}, start_index=0, end_index=10, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "not an integer"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value="not an integer", start_index=0, end_index=15, content="")
    field = Field()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            if key == "user":
                return MockToken(value={"age": "invalid"}, start_index=10, end_index=30, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"user": {"age": "invalid"}}, start_index=0, end_index=40, content="")
    fields = {"user": Schema(fields={"age": Field()})}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.index == ["user", "age"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"name": "John"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"name": "John"}, start_index=0, end_index=20, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value="invalid", start_index=0, end_index=7, content="")
    union = Union(any_of=[Field(), Field()])
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "union"
        assert msg.start_position is not None
        assert msg.end_position is not None


# LLM-generated content at query #13
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 2
    assert message.end_position.char_index == 1

def test_validate_with_positions_schema_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={1: "value"}, start_index=0, end_index=10, content="{1: 'value'}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid_key"
    assert message.index == [1]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 11
    assert message.end_position.char_index == 10

def test_validate_with_positions_schema_nested_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"max_length": "Must have at most {max_length} characters."}
        def __init__(self, max_length):
            super().__init__()
            self.max_length = max_length
        def validate(self, value):
            if len(value) > self.max_length:
                raise self.validation_error("max_length")
            return value
    fields = {"user": Schema(fields={"name": MockField(max_length=5)})}
    schema = Schema(fields=fields)
    token = Token(value={"user": {"name": "longname"}}, start_index=0, end_index=25, content="{'user': {'name': 'longname'}}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "max_length"
    assert message.index == ["user", "name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 26
    assert message.end_position.char_index == 25

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntegerField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[StringField(), IntegerField()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "union"
    assert message.index == []
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 4
    assert message.end_position.char_index == 3

def test_validate_with_positions_successful_validation():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content="{'name': 'test'}")
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"min_length": "Must have at least {min_length} characters."}
        def __init__(self, min_length):
            super().__init__()
            self.min_length = min_length
        def validate(self, value):
            if len(value) < self.min_length:
                raise self.validation_error("min_length")
            return value
    fields = {"a": MockField(min_length=10), "b": MockField(min_length=10)}
    schema = Schema(fields=fields)
    token = Token(value={"a": "short", "b": "also"}, start_index=0, end_index=25, content="{'a': 'short', 'b': 'also'}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].index == ["a"]
    assert messages[1].index == ["b"]
    assert messages[0].start_position.char_index < messages[1].start_position.char_index


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def _get_value(self):
            return self._value
        
        def _get_child_token(self, key):
            return MockToken(value={}, start_index=0, end_index=0, content="")
        
        def _get_key_token(self, key):
            return MockToken(value=key, start_index=0, end_index=0, content="")

    class MockField(Field):
        errors = {"required": "This field is required."}
        
        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required")

    field = MockField()
    schema = Schema(fields={"field": field})
    token = MockToken(value={}, start_index=0, end_index=0, content="")
    result = validate_with_positions(token=token, validator=schema)
    assert False


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    import typesystem.tokenize.positional_validation as module

    class MockToken(Token):
        def _get_value(self):
            return {"existing": "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content=" " * 30)
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=15, content=" " * 30)

    class MockField(Field):
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Required field", code="required", index=["missing"])])

    schema = Schema(fields={"missing": MockField()})
    token = MockToken(value={"existing": "value"}, start_index=0, end_index=30, content=" " * 30)
    try:
        module.validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].start_position is not None
        assert messages[0].end_position is not None


# LLM-generated content at query #16
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            if value is None:
                raise self.validation_error("null")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'name' is required."
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 12
        assert msg.end_position.char_index == 11

def test_validate_with_positions_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={1: "value"}, start_index=0, end_index=10, content='{1: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "All object keys must be strings."
        assert msg.code == "invalid_key"
        assert msg.index == [1]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 2
        assert msg.start_position.char_index == 1
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 2
        assert msg.end_position.char_index == 1

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            if value is None:
                raise self.validation_error("null")
            return value
    nested_fields = {"inner": MockField()}
    nested_schema = Schema(fields=nested_fields)
    fields = {"outer": nested_schema}
    schema = Schema(fields=fields)
    token = Token(value={"outer": {}}, start_index=0, end_index=15, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'inner' is required."
        assert msg.code == "required"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 11
        assert msg.start_position.char_index == 10
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 12
        assert msg.end_position.char_index == 11

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"field": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"field": "value"}, start_index=0, end_index=16, content='{"field": "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Custom error."
        assert msg.code == "custom"
        assert msg.index == ["field"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 10
        assert msg.start_position.char_index == 9
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 17
        assert msg.end_position.char_index == 16

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value.upper()
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "TEST"}

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields, allow_null=True)
    token = Token(value=None, start_index=0, end_index=4, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields, allow_null=False)
    token = Token(value=None, start_index=0, end_index=4, content="null")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "May not be null."
        assert msg.code == "null"
        assert msg.index == []
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 5
        assert msg.end_position.char_index == 4

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=12, content='{"a":1,"b":2}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        msg1 = messages[0]
        msg2 = messages[1]
        assert msg1.index == ["a"]
        assert msg2.index == ["b"]
        assert msg1.start_position.char_index == 5
        assert msg2.start_position.char_index == 10
        assert msg1.start_position.char_index < msg2.start_position.char_index

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
   


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    msg = error.messages[0]
    assert msg.text == "The field 'name' is required."
    assert msg.code == "required"
    assert msg.index == ["name"]

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    inner_schema = Schema(fields={"count": MockField()})
    outer_schema = Schema(fields={"inner": inner_schema})
    token = Token(value={"inner": {"count": "invalid"}}, start_index=0, end_index=30, content='{"inner": {"count": "invalid"}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    msg = error.messages[0]
    assert msg.code == "type"
    assert msg.index == ["inner", "count"]

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def __init__(self, error_code="custom", **kwargs):
            super().__init__(**kwargs)
            self.error_code = error_code
        def validate(self, value):
            raise self.validation_error(self.error_code)
    fields = {"a": MockField(error_code="error_a"), "b": MockField(error_code="error_b")}
    schema = Schema(fields=fields)
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=20, content='{"a": 1, "b": 2}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 2
    codes = [msg.code for msg in error.messages]
    assert "error_a" in codes
    assert "error_b" in codes

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field, Union
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class IntField(Field):
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    class StrField(Field):
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    union_field = Union(any_of=[IntField(), StrField()])
    schema = Schema(fields={"data": union_field})
    token = Token(value={"data": None}, start_index=0, end_index=15, content='{"data": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    msg = error.messages[0]
    assert msg.code == "union" or msg.code == "null"


# LLM-generated content at query #2
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockToken(Token):
        def _get_value(self):
            return {"existing": "value"}
        def _get_child_token(self, key):
            if key == "existing":
                return MockToken(value="value", start_index=10, end_index=14, content='{"existing": "value"}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value={"existing": "value"}, start_index=0, end_index=20, content='{"existing": "value"}')
    fields = {"existing": Field(), "missing": Field()}
    validator = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except typesystem.exceptions.ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "The field 'missing' is required."
        assert msg.code == "required"
        assert msg.index == ["missing"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 21
        assert msg.end_position.char_index == 20

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockToken(Token):
        def _get_value(self):
            return {"outer": {}}
        def _get_child_token(self, key):
            if key == "outer":
                return MockToken(value={}, start_index=10, end_index=11, content='{"outer": {}}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value={"outer": {}}, start_index=0, end_index=15, content='{"outer": {}}')
    inner_schema = Schema(fields={"inner": Field()})
    validator = Schema(fields={"outer": inner_schema})
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except typesystem.exceptions.ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "The field 'inner' is required."
        assert msg.code == "required"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 12
        assert msg.start_position.char_index == 11
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 12
        assert msg.end_position.char_index == 11

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    class MockToken(Token):
        def _get_value(self):
            return {"field": "invalid"}
        def _get_child_token(self, key):
            if key == "field":
                return MockToken(value="invalid", start_index=10, end_index=17, content='{"field": "invalid"}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value={"field": "invalid"}, start_index=0, end_index=20, content='{"field": "invalid"}')
    validator = Schema(fields={"field": MockField()})
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except typesystem.exceptions.ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "Custom error."
        assert msg.code == "custom"
        assert msg.index == ["field"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 11
        assert msg.start_position.char_index == 10
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 18
        assert msg.end_position.char_index == 17

def test_validate_with_positions_successful_validation():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field": "valid"}
        def _get_child_token(self, key):
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value={"field": "valid"}, start_index=0, end_index=18, content='{"field": "valid"}')
    validator = Schema(fields={"field": Field()})
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"field": "valid"}

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockToken(Token):
        def _get_value(self):
            return {"b": "value", "a": "value"}
        def _get_child_token(self, key):
            if key == "a":
                return MockToken(value="value", start_index=10, end_index=15, content='{"a": "value", "b": "value"}')
            if key == "b":
                return MockToken(value="value", start_index=20, end_index=25, content='{"a": "value", "b": "value"}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value={"b": "value", "a": "value"}, start_index=0, end_index=30, content='{"a": "value", "b": "value"}')
    fields = {"a": Field(), "b": Field(), "c": Field()}
    validator = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'c' is required."
        assert msg.index == ["c"]

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    token = MockToken(value="invalid", start_index=0, end_index=7, content='"invalid"')
    validator = Union(any_of=[Field()])
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except typesystem.exceptions.ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.text == "Did not match any valid type."
        assert msg.code == "union"
        assert msg.index == []
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 8
        assert msg.end_position.char_index == 7


# LLM-generated content at query #3
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'name' is required."
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 12
    assert message.end_position.char_index == 11

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": 123}, start_index=0, end_index=12, content='{"name": 123}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be a string."
    assert message.code == "type"
    assert message.index == ["name"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 13
    assert message.end_position.char_index == 12

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_fields = {"city": MockField()}
    inner_schema = Schema(fields=inner_fields)
    outer_fields = {"address": inner_schema}
    outer_schema = Schema(fields=outer_fields)
    token = Token(value={"address": {}}, start_index=0, end_index=15, content='{"address": {}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'city' is required."
    assert message.code == "required"
    assert message.index == ["address", "city"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 16
    assert message.end_position.char_index == 15

def test_validate_with_positions_nested_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_fields = {"city": MockField()}
    inner_schema = Schema(fields=inner_fields)
    outer_fields = {"address": inner_schema}
    outer_schema = Schema(fields=outer_fields)
    token = Token(value={"address": {"city": 456}}, start_index=0, end_index=24, content='{"address": {"city": 456}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be a string."
    assert message.code == "type"
    assert message.index == ["address", "city"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 25
    assert message.end_position.char_index == 24

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField(), "email": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": 123, "email": 456}, start_index=0, end_index=25, content='{"name": 123, "email": 456}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 2
    first_message = error.messages[0]
    second_message = error.messages[1]
    assert first_message.index == ["name"]
    assert second_message.index == ["email"]
    assert first_message.start_position.char_index < second_message.start_position.char_index

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field, Union
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union_field = StringField() | IntField()
    fields = {"data": union_field}
    schema = Schema(fields=fields)
    token = Token(value={"data": None}, start_index=0, end_index=14, content='{"data": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.code == "union"
    assert message.index == ["data"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 15
    assert message.end_position.char_index == 14

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "John"}, start_index=0, end


# LLM-generated content at query #4
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        module.validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 2
        assert msg.end_position.char_index == 1


def test_validate_with_positions_schema_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={1: "value"}, start_index=0, end_index=10, content="{1: 'value'}")
    try:
        module.validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid_key"
        assert msg.index == [1]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 11
        assert msg.end_position.char_index == 10


def test_validate_with_positions_schema_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class MockField(Field):
        errors = {"custom": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("custom")
    inner_schema = Schema(fields={"inner": MockField()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": "bad"}}, start_index=0, end_index=25, content='{"outer": {"inner": "bad"}}')
    try:
        module.validate_with_positions(token=token, validator=outer_schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "custom"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 26
        assert msg.end_position.char_index == 25


def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    field = MockField()
    token = Token(value=123, start_index=5, end_index=7, content='  "123"')
    try:
        module.validate_with_positions(token=token, validator=field)
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "type"
        assert msg.index == []
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 3
        assert msg.start_position.char_index == 2
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 8
        assert msg.end_position.char_index == 7


def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[StringField(), IntField()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        module.validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "union"
        assert msg.index == []
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 4
        assert msg.end_position.char_index == 3


def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    import typesystem.tokenize.positional_validation as module
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "test"}, start_index=0, end_index=15, content='{"name": "test"}')
    result = module.validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.fields import Field, Schema
    from typesystem.base import Message
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._child_tokens = {}
            self._key_tokens = {}

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return self._child_tokens.get(key, MockToken(None, 0, 0))

        def _get_key_token(self, key):
            return self._key_tokens.get(key, MockToken(None, 0, 0))

        def lookup(self, index):
            if not index:
                return self
            token = self
            for key in index:
                token = token._get_child_token(key)
            return token

    class MockField(Field):
        errors = {"required": "This field is required."}

        def validate(self, value):
            raise self.validation_error("required")

    schema = Schema(fields={"field": MockField()})
    token = MockToken(value={}, content='{"field": null}')
    token._child_tokens = {"field": MockToken(value=None, start_index=10, end_index=13, content='{"field": null}')}
    token._key_tokens = {"field": MockToken(value="field", start_index=2, end_index=6, content='{"field": null}')}

    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.text == "The field 'field' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #6
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise self.validation_error("required")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'field' is required."
    assert message.code == "required"
    assert message.index == ["field"]
    assert message.start_position is not None
    assert message.end_position is not None

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise self.validation_error("required")
    inner_schema = Schema(fields={"inner": MockField()})
    schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": None}}, start_index=0, end_index=20, content='{"outer": {"inner": null}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]
    assert message.start_position is not None
    assert message.end_position is not None

def test_validate_with_positions_custom_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Custom error"}
        def validate(self, value):
            raise self.validation_error("custom")
    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": "invalid"}, start_index=0, end_index=15, content='{"field": "invalid"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Custom error"
    assert message.code == "custom"
    assert message.index == ["field"]
    assert message.start_position is not None
    assert message.end_position is not None

def test_validate_with_positions_multiple_errors():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        def validate(self, value):
            raise self.validation_error("required")
    class MockField2(Field):
        errors = {"invalid": "Invalid value"}
        def validate(self, value):
            raise self.validation_error("invalid")
    schema = Schema(fields={"field1": MockField1(), "field2": MockField2()})
    token = Token(value={"field1": None, "field2": "bad"}, start_index=0, end_index=25, content='{"field1": null, "field2": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 2
    messages = error.messages()
    required_message = next(m for m in messages if m.code == "required")
    invalid_message = next(m for m in messages if m.code == "invalid")
    assert required_message.text == "The field 'field1' is required."
    assert required_message.index == ["field1"]
    assert invalid_message.text == "Invalid value"
    assert invalid_message.index == ["field2"]
    assert all(m.start_position is not None and m.end_position is not None for m in messages)

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": "valid"}, start_index=0, end_index=15, content='{"field": "valid"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "valid"}

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        errors = {"type": "Must be integer"}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    class MockField2(Field):
        errors = {"type": "Must be string"}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[MockField1(), MockField2()])
    token = Token(value=True, start_index=0, end_index=4, content="true")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "union"
    assert message.start_position is not None
    assert message.end_position is not None

def test_validate_with_positions_sorted_by_position():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise self.validation_error("required")
    schema = Schema(fields={"field1": MockField(), "field2": MockField()})
    token = Token(value={"field2": None, "field1": None}, start_index=0, end_index=30, content='{"field2": null, "field1": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].index == ["field2"]
    assert messages[1].index == ["field1"]
    assert messages[0].start_position.char_index < messages[1].start_position.char_index


# LLM-generated content at query #7
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self
    token = MockToken(value={}, start_index=0, end_index=10, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'name' is required."
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_fields = {"inner_name": MockField()}
    inner_schema = Schema(fields=inner_fields)
    fields = {"outer": inner_schema}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content, child_map=None):
            super().__init__(value, start_index, end_index, content)
            self.child_map = child_map or {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self.child_map.get(key, self)
        def _get_key_token(self, key):
            return self.child_map.get(key, self)
    outer_token = MockToken(value={"outer": {}}, start_index=0, end_index=20, content='{"outer": {}}')
    inner_token = MockToken(value={}, start_index=9, end_index=11, content='{"outer": {}}')
    outer_token.child_map = {"outer": inner_token}
    try:
        validate_with_positions(token=outer_token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'inner_name' is required."
        assert msg.code == "required"
        assert msg.index == ["outer", "inner_name"]
        assert msg.start_position == inner_token.start
        assert msg.end_position == inner_token.end

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content, child_map=None):
            super().__init__(value, start_index, end_index, content)
            self.child_map = child_map or {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self.child_map.get(key, self)
        def _get_key_token(self, key):
            return self.child_map.get(key, self)
    root_value = {"name": 123}
    root_token = MockToken(value=root_value, start_index=0, end_index=20, content='{"name": 123}')
    child_token = MockToken(value=123, start_index=10, end_index=12, content='{"name": 123}')
    root_token.child_map = {"name": child_token}
    try:
        validate_with_positions(token=root_token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Must be a string."
        assert msg.code == "type"
        assert msg.index == ["name"]
        assert msg.start_position == child_token.start
        assert msg.end_position == child_token.end

def test_validate_with_positions_successful_validation():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def _get_value(self):
            return {"name": "valid"}
        def _get_child_token(self, key):
            return self
        def _get_key_token(self, key):
            return self
    token = MockToken(value={"name": "valid"}, start_index=0, end_index=20, content='{"name": "valid"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "valid"}

def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content, child_map=None):
            super().__init__(value, start_index, end_index, content)
            self.child_map = child_map or {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self.child_map.get(key, self)
        def _get_key_token(self, key):
            return self.child_map.get(key, self)
    root_value = {"a": 1, "b": 2}
    root_token = MockToken(value=root_value, start_index=0, end_index=30, content='{"a": 1, "b": 2}')
    token_a = MockToken(value=1, start_index=10, end_index=10, content='{"a": 1, "b": 2}')
    token_b = MockToken(value=2, start_index=20, end_index=20, content='{"a": 1, "b": 2}')
    root_token.child_map = {"a": token_a, "b": token_b}
    try:
        validate_with_positions(token=root_token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 10
        assert messages[1].start_position.char_index == 20

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union_field = Union(any_of=[StringField(), IntField()])
    class MockToken(Token):
        def


# LLM-generated content at query #8
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={}, start_index=0, end_index=10, content="")
    field = Field()
    schema = Schema(fields={"field1": field})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'field1' is required."
    assert message.code == "required"
    assert message.index == ["field1"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 1
    assert message.end_position.char_index == 10

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field1": "not an int"}
        def _get_child_token(self, key):
            if key == "field1":
                return MockToken(value="not an int", start_index=8, end_index=18, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"field1": "not an int"}, start_index=0, end_index=20, content="")
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"field1": IntField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be an integer."
    assert message.code == "type"
    assert message.index == ["field1"]
    assert message.start_position.char_index == 8
    assert message.end_position.char_index == 18

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"outer": {}}
        def _get_child_token(self, key):
            if key == "outer":
                return MockToken(value={}, start_index=8, end_index=10, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"outer": {}}, start_index=0, end_index=12, content="")
    inner_schema = Schema(fields={"inner": Field()})
    schema = Schema(fields={"outer": inner_schema})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]
    assert message.start_position.char_index == 8
    assert message.end_position.char_index == 10

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field2": "invalid", "field1": "invalid"}
        def _get_child_token(self, key):
            if key == "field1":
                return MockToken(value="invalid", start_index=10, end_index=17, content="")
            if key == "field2":
                return MockToken(value="invalid", start_index=0, end_index=7, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"field2": "invalid", "field1": "invalid"}, start_index=0, end_index=20, content="")
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"field1": IntField(), "field2": IntField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 2
    first_message = error.messages[0]
    second_message = error.messages[1]
    assert first_message.start_position.char_index == 0
    assert second_message.start_position.char_index == 10

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "not an int or string"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value="not an int or string", start_index=0, end_index=20, content="")
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[IntField(), StringField()])
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Did not match any valid type."
    assert message.code == "union"
    assert message.index == []
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 20


# LLM-generated content at query #9
#--------------------------

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    field = MockField()
    token = Token(value="valid", start_index=0, end_index=4, content="valid")
    result = validate_with_positions(token=token, validator=field)
    assert result == "valid"

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"custom": "Invalid value"}
        def validate(self, value):
            raise self.validation_error("custom")
    field = MockField()
    token = Token(value="invalid", start_index=0, end_index=6, content="invalid")
    try:
        validate_with_positions(token=token, validator=field)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid value"
        assert msg.code == "custom"
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_schema_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'name' is required."
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_schema_nested_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        def validate(self, value):
            return value
    inner_schema = Schema(fields={"inner": MockField()})
    schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {}}, start_index=0, end_index=12, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=schema)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'inner' is required."
        assert msg.code == "required"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position.char_index > token.start.char_index
        assert msg.end_position.char_index < token.end.char_index

def test_validate_with_positions_schema_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"custom": "Field error"}
        def validate(self, value):
            raise self.validation_error("custom")
    schema = Schema(fields={"a": MockField(), "b": MockField()})
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=11, content='{"a":1,"b":2}')
    try:
        validate_with_positions(token=token, validator=schema)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index < messages[1].start_position.char_index

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field, Union
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockFieldA(Field):
        errors = {"type": "Must be string"}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class MockFieldB(Field):
        errors = {"type": "Must be integer"}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[MockFieldA(), MockFieldB()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "union"
        assert msg.start_position == token.start
        assert msg.end_position == token.end


# LLM-generated content at query #10
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    schema = Schema(fields={"field": MockField()})
    token = MockToken(value={}, start_index=0, end_index=0, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.start_position is not None
        assert message.end_position is not None
        assert message.text == "The field 'field' is required."


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockField(Field):
        errors = {"required": "This field is required."}

        def validate(self, value):
            raise self.validation_error("required")

    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert any(msg.code == "required" for msg in e.messages())


# LLM-generated content at query #12
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.text == "The field 'name' is required."
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 10

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": 123}, start_index=0, end_index=12, content='{"name": 123}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.index == ["name"]
        assert msg.text == "Must be a string."
        assert msg.start_position.char_index == 9
        assert msg.end_position.char_index == 11

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_fields = {"city": MockField()}
    inner_schema = Schema(fields=inner_fields)
    outer_fields = {"address": inner_schema}
    outer_schema = Schema(fields=outer_fields)
    token = Token(value={"address": {}}, start_index=0, end_index=15, content='{"address": {}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["address", "city"]
        assert msg.text == "The field 'city' is required."
        assert msg.start_position.char_index == 12
        assert msg.end_position.char_index == 12

def test_validate_with_positions_nested_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_fields = {"city": MockField()}
    inner_schema = Schema(fields=inner_fields)
    outer_fields = {"address": inner_schema}
    outer_schema = Schema(fields=outer_fields)
    token = Token(value={"address": {"city": 456}}, start_index=0, end_index=24, content='{"address": {"city": 456}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.index == ["address", "city"]
        assert msg.text == "Must be a string."
        assert msg.start_position.char_index == 21
        assert msg.end_position.char_index == 23

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField(), "email": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": 123, "email": 456}, start_index=0, end_index=25, content='{"name": 123, "email": 456}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        msg1 = messages[0]
        msg2 = messages[1]
        assert msg1.code == "type"
        assert msg1.index == ["name"]
        assert msg1.start_position.char_index == 9
        assert msg1.end_position.char_index == 11
        assert msg2.code == "type"
        assert msg2.index == ["email"]
        assert msg2.start_position.char_index == 22
        assert msg2.end_position.char_index == 24

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class StringField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[StringField(), IntField()])
    token = Token(value=True, start_index=0, end_index=4, content="true")
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "union"
        assert msg.index == []
        assert msg.text == "Did not match any valid type."
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 3

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "John"}, start_index=0, end_index=14, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField(allow_null=True)}
    schema = Schema(fields=fields, allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None


# LLM-generated content at query #13
#--------------------------

def test_validate_with_positions_required_field_error():
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required")
    class MockSchema(Schema):
        def __init__(self):
            super().__init__(fields={"field": MockField()})
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=15, content="")
    token = MockToken(value={}, start_index=0, end_index=30, content="")
    validator = MockSchema()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'field' is required."
        assert message.index == ["field"]
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #14
#--------------------------

def test_validate_with_positions_schema_required_field():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 1

def test_validate_with_positions_schema_invalid_key():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    token = Token(value={1: "value"}, start_index=0, end_index=10, content="{1: 'value'}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid_key"
        assert msg.index == [1]
        assert msg.start_position.char_index == 1
        assert msg.end_position.char_index == 1

def test_validate_with_positions_schema_field_validation_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class CustomField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"name": CustomField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "invalid"}, start_index=0, end_index=15, content="{'name': 'invalid'}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "custom"
        assert msg.index == ["name"]
        assert msg.start_position.char_index == 9
        assert msg.end_position.char_index == 14

def test_validate_with_positions_union_error():
    from typesystem.fields import Union, Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class FieldA(Field):
        errors = {"type": "Must be string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    class FieldB(Field):
        errors = {"type": "Must be integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    union = Union(any_of=[FieldA(), FieldB()])
    token = Token(value=3.14, start_index=0, end_index=3, content="3.14")
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "union"
        assert msg.index == []
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 3

def test_validate_with_positions_schema_nested_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class CustomField(Field):
        errors = {"custom": "Nested error."}
        def validate(self, value):
            raise self.validation_error("custom")
    inner_schema = Schema(fields={"inner": CustomField()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": "value"}}, start_index=0, end_index=30, content="{'outer': {'inner': 'value'}}")
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "custom"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position.char_index == 19
        assert msg.end_position.char_index == 24

def test_validate_with_positions_success():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "John"}, start_index=0, end_index=15, content="{'name': 'John'}")
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_null_allowed():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    fields = {"name": Field(allow_null=True)}
    schema = Schema(fields=fields, allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_schema_type_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    token = Token(value="not an object", start_index=0, end_index=12, content="not an object")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "type"
        assert msg.index == []
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 12

def test_validate_with_positions_messages_sorted_by_position():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class ErrorField(Field):
        errors = {"error": "Error."}
        def validate(self, value):
            raise self.validation_error("error")
    fields = {"a": ErrorField(), "b": ErrorField()}
    schema = Schema(fields=fields)
    token = Token(value={"b": "val1", "a": "val2"}, start_index=0, end_index=20, content="{'b': 'val1', 'a': 'val2'}")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        msgs = e.messages()
        assert len(msgs) == 2
        assert msgs[0].index == ["b"]
        assert msgs[1].index == ["a"]
        assert msgs[0].start_position.char_index < msgs[1].start_position.char_index


