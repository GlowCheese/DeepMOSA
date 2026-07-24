####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={"age": 25}, start_index=0, end_index=10, content='{"age": 25}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.text == "The field 'name' is required."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_type():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={"name": 123}, start_index=0, end_index=12, content='{"name": 123}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.index == ["name"]
        assert msg.text == "Must be a string."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_nested_required():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    inner_schema = Schema(fields={"inner_name": MockField()})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {}}, start_index=0, end_index=12, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["outer", "inner_name"]
        assert msg.text == "The field 'inner_name' is required."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": MockField(), "title": MockField()})
    token = Token(value={"name": 123, "title": 456}, start_index=0, end_index=24, content='{"name": 123, "title": 456}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        msg1, msg2 = messages
        assert msg1.code == "type"
        assert msg1.index == ["name"]
        assert msg2.code == "type"
        assert msg2.index == ["title"]
        assert msg1.start_position.char_index <= msg2.start_position.char_index

def test_validate_with_positions_valid():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    schema = Schema(fields={"name": MockField()})
    token = Token(value={"name": "test"}, start_index=0, end_index=14, content='{"name": "test"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_null_allowed():
    from typesystem.fields import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    schema = Schema(fields={}, allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.fields import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    schema = Schema(fields={}, allow_null=False)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "null"
        assert msg.text == "May not be null."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_key():
    from typesystem.fields import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
    schema = Schema(fields={})
    token = Token(value={123: "value"}, start_index=0, end_index=12, content='{123: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "invalid_key"
        assert msg.index == [123]
        assert msg.text == "All object keys must be strings."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_union_field_error():
    from typesystem.fields import Field, Union, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    import typesystem
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
    schema = Schema(fields={"data": union_field})
    token = Token(value={"data": 3.14}, start_index=0, end_index=14, content='{"data": 3.14}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "union"
        assert msg.index == ["data"]
        assert msg.text == "Did not match any valid type."
        assert msg.start_position is not None
        assert msg.end_position is not None


# LLM-generated content at query #2
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    token._get_value = lambda: {"field": None}
    token._get_child_token = lambda key: Token(value=None, start_index=8, end_index=11, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'field' is required."
    assert message.code == "required"
    assert message.index == ["field"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 1

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    inner_schema = Schema(fields={"inner": MockField()})
    schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"inner": None}}, start_index=0, end_index=20, content='{"outer": {"inner": null}}')
    token._get_value = lambda: {"outer": {"inner": None}}
    token._get_child_token = lambda key: Token(value={"inner": None}, start_index=10, end_index=20, content='{"outer": {"inner": null}}')
    token._get_child_token("outer")._get_child_token = lambda key: Token(value=None, start_index=18, end_index=21, content='{"outer": {"inner": null}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 1

def test_validate_with_positions_custom_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": "invalid"}, start_index=0, end_index=20, content='{"field": "invalid"}')
    token._get_value = lambda: {"field": "invalid"}
    token._get_child_token = lambda key: Token(value="invalid", start_index=10, end_index=18, content='{"field": "invalid"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Custom error."
    assert message.code == "custom"
    assert message.index == ["field"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 1

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
    schema = Schema(fields={"field1": MockField(), "field2": MockField()})
    token = Token(value={"field1": "invalid1", "field2": "invalid2"}, start_index=0, end_index=40, content='{"field1": "invalid1", "field2": "invalid2"}')
    token._get_value = lambda: {"field1": "invalid1", "field2": "invalid2"}
    token._get_child_token = lambda key: Token(value="invalid1" if key == "field1" else "invalid2", start_index=12 if key == "field1" else 34, end_index=20 if key == "field1" else 42, content='{"field1": "invalid1", "field2": "invalid2"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 2
    message1 = error.messages[0]
    message2 = error.messages[1]
    assert message1.index == ["field1"]
    assert message2.index == ["field2"]
    assert message1.start_position.char_index < message2.start_position.char_index

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            raise self.validation_error("type")
    class MockField2(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            raise self.validation_error("type")
    union = Union(any_of=[MockField1(), MockField2()])
    token = Token(value="invalid", start_index=0, end_index=6, content='"invalid"')
    token._get_value = lambda: "invalid"
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.code == "union"
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 6

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": "valid"}, start_index=0, end_index=18, content='{"field": "valid"}')
    token._get_value = lambda: {"field": "valid"}
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "valid"}


# LLM-generated content at query #3
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
    schema = Schema(fields={"name": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.text == "The field 'name' is required."

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
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value="not an object", start_index=0, end_index=15, content="")
    schema = Schema(fields={})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "type"
        assert msg.index == []
        assert msg.text == "Must be an object."

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"user": {}}
        def _get_child_token(self, key):
            if key == "user":
                return MockToken(value={}, start_index=8, end_index=12, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"user": {}}, start_index=0, end_index=20, content="")
    schema = Schema(fields={"user": Schema(fields={"name": Field()})})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["user", "name"]
        assert msg.text == "The field 'name' is required."

def test_validate_with_positions_invalid_key():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {123: "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={123: "value"}, start_index=0, end_index=20, content="")
    schema = Schema(fields={})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid_key"
        assert msg.index == [123]
        assert msg.text == "All object keys must be strings."

def test_validate_with_positions_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"max_length": "Must have at most {max_length} characters."}
        def validate(self, value):
            if len(value) > 5:
                raise self.validation_error("max_length")
            return value
    class MockToken(Token):
        def _get_value(self):
            return {"name": "too long"}
        def _get_child_token(self, key):
            if key == "name":
                return MockToken(value="too long", start_index=10, end_index=18, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"name": "too long"}, start_index=0, end_index=30, content="")
    schema = Schema(fields={"name": MockField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "max_length"
        assert msg.index == ["name"]
        assert msg.text == "Must have at most 5 characters."

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"a": "x", "b": "y"}
        def _get_child_token(self, key):
            if key == "a":
                return MockToken(value="x", start_index=5, end_index=6, content="")
            if key == "b":
                return MockToken(value="y", start_index=12, end_index=13, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"a": "x", "b": "y"}, start_index=0, end_index=20, content="")
    class FailingField(Field):
        errors = {"always": "Always fails."}
        def validate(self, value):
            raise self.validation_error("always")
    schema = Schema(fields={"a": FailingField(), "b": FailingField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        msgs = e.messages()
        assert len(msgs) == 2
        assert msgs[0].start_position.char_index == 5
        assert msgs[1].start_position.char_index == 12

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return None
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value=None, start_index=0, end_index=4, content="")
    schema = Schema(fields={}, allow_null=True)
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return None
        def _get_child_token(self, key):
            return MockToken(value=None,


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value

    schema = Schema(fields={"field": MockField()})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert isinstance(e, ValidationError)
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["field"]
        assert msg.start_position is not None
        assert msg.end_position is not None


# LLM-generated content at query #5
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
    content = '{"age": 25}'
    token = Token(value={"age": 25}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.text == "The field 'name' is required."
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == len(content)-1

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
    content = '{"name": 123}'
    token = Token(value={"name": 123}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "type"
    assert message.index == ["name"]
    assert message.text == "Must be a string."
    assert message.start_position.char_index == content.find("123")
    assert message.end_position.char_index == content.find("123") + 2

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
    content = '{"address": {}}'
    token = Token(value={"address": {}}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["address", "city"]
    assert message.text == "The field 'city' is required."
    assert message.start_position.char_index == content.find("{}") + 1
    assert message.end_position.char_index == content.find("{}") + 1

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
    content = '{"name": 123, "email": 456}'
    token = Token(value={"name": 123, "email": 456}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 2
    messages = error.messages()
    assert messages[0].code == "type"
    assert messages[0].index == ["name"]
    assert messages[0].start_position.char_index == content.find("123")
    assert messages[0].end_position.char_index == content.find("123") + 2
    assert messages[1].code == "type"
    assert messages[1].index == ["email"]
    assert messages[1].start_position.char_index == content.find("456")
    assert messages[1].end_position.char_index == content.find("456") + 2

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field, Union
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
    fields = {"value": union_field}
    schema = Schema(fields=fields)
    content = '{"value": null}'
    token = Token(value={"value": None}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "null"
    assert message.index == ["value"]
    assert message.start_position.char_index == content.find("null")
    assert message.end_position.char_index == content.find("null") + 3

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
    content = '{"name": "John"}'
    token = Token(value={"name": "John"}, start_index=0, end_index=len(content)-1, content=content)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_invalid_key():
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
    content = '{123: "value"}'
    token = Token(value={123: "value"}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid_key"
    assert message.index == [123]
    assert message.start_position.char_index == content.find("123")
    assert message.end_position.char_index == content.find("123") + 2


# LLM-generated content at query #6
#--------------------------

def test_validate_with_positions_success():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    token = Token(value={"key": "value"}, start_index=0, end_index=10, content='{"key": "value"}')
    validator = MockField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"key": "value"}

def test_validate_with_positions_required_error():
    from typesystem.fields import Schema, Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        pass
    fields = {"required_field": MockField()}
    validator = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=2, content="{}")
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'required_field' is required."
        assert message.code == "required"
        assert message.index == ["required_field"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 2

def test_validate_with_positions_nested_required_error():
    from typesystem.fields import Schema, Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        pass
    inner_fields = {"inner_required": MockField()}
    inner_schema = Schema(fields=inner_fields)
    outer_fields = {"outer_key": inner_schema}
    validator = Schema(fields=outer_fields)
    token = Token(value={"outer_key": {}}, start_index=0, end_index=20, content='{"outer_key": {}}')
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'inner_required' is required."
        assert message.code == "required"
        assert message.index == ["outer_key", "inner_required"]
        assert message.start_position.char_index == 13
        assert message.end_position.char_index == 15

def test_validate_with_positions_general_error():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    validator = MockField()
    token = Token(value="bad", start_index=0, end_index=5, content='"bad"')
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "Custom error."
        assert message.code == "custom"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 5

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.fields import Schema, Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"field1": MockField(), "field2": MockField()}
    validator = Schema(fields=fields)
    token = Token(value={"field1": "x", "field2": "y"}, start_index=0, end_index=30, content='{"field1": "x", "field2": "y"}')
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        first_message = messages[0]
        second_message = messages[1]
        assert first_message.start_position.char_index < second_message.start_position.char_index
        assert first_message.index == ["field1"]
        assert second_message.index == ["field2"]

def test_validate_with_positions_invalid_key_error():
    from typesystem.fields import Schema, Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    class MockField(Field):
        pass
    fields = {"valid": MockField()}
    validator = Schema(fields=fields)
    token = Token(value={123: "invalid"}, start_index=0, end_index=20, content='{123: "invalid"}')
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "invalid_key"
        assert message.index == [123]
        assert message.start_position.char_index == 1
        assert message.end_position.char_index == 4


# LLM-generated content at query #7
#--------------------------

def test_validate_with_positions_success():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    field = MockField()
    token = Token(value="test", start_index=0, end_index=3, content="test")
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"

def test_validate_with_positions_validation_error_without_index():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    class MockField(Field):
        errors = {"custom": "Invalid value"}
        def validate(self, value):
            raise self.validation_error("custom")
    field = MockField()
    token = Token(value="bad", start_index=0, end_index=2, content="bad")
    try:
        validate_with_positions(token=token, validator=field)
        assert False
    except Exception as e:
        assert isinstance(e, ValidationError)
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid value"
        assert msg.code == "custom"
        assert msg.index == []
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_validation_error_with_index():
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    class MockField(Field):
        errors = {"custom": "Invalid value"}
        def validate(self, value):
            error = self.validation_error("custom")
            error._messages = [Message(text="Invalid value", code="custom", index=["key"])]
            raise error
    field = MockField()
    token = Token(value={"key": "bad"}, start_index=0, end_index=10, content='{"key":"bad"}')
    try:
        validate_with_positions(token=token, validator=field)
        assert False
    except Exception as e:
        assert isinstance(e, ValidationError)
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid value"
        assert msg.code == "custom"
        assert msg.index == ["key"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 8
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 11

def test_validate_with_positions_required_error():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    class MockField(Field):
        def validate(self, value):
            return value
    schema = Schema(fields={"required_field": MockField()})
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
        assert False
    except Exception as e:
        assert isinstance(e, ValidationError)
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'required_field' is required."
        assert msg.code == "required"
        assert msg.index == ["required_field"]
        assert msg.start_position == token.start
        assert msg.end_position == token.end

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    class MockField(Field):
        errors = {"custom": "Invalid"}
        def validate(self, value):
            raise self.validation_error("custom")
    schema = Schema(fields={"field1": MockField(), "field2": MockField()})
    token = Token(value={"field1": "bad", "field2": "bad"}, start_index=0, end_index=30, content='{"field1":"bad","field2":"bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
        assert False
    except Exception as e:
        assert isinstance(e, ValidationError)
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index < messages[1].start_position.char_index


# LLM-generated content at query #8
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
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
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.text == "The field 'name' is required."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_type_error():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
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
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "type"
        assert msg.index == ["name"]
        assert msg.text == "Must be a string."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_nested_required():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
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
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["address", "city"]
        assert msg.text == "The field 'city' is required."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
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
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        msg1 = messages[0]
        msg2 = messages[1]
        assert msg1.code == "type"
        assert msg1.index == ["name"]
        assert msg2.code == "type"
        assert msg2.index == ["email"]
        assert msg1.start_position.char_index < msg2.start_position.char_index

def test_validate_with_positions_valid():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": "John"}, start_index=0, end_index=15, content='{"name": "John"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "John"}

def test_validate_with_positions_null_allowed():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields, allow_null=True)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields, allow_null=False)
    token = Token(value=None, start_index=0, end_index=3, content="null")
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "null"
        assert msg.text == "May not be null."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_key():
    from typesystem.fields import Field, Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={123: "John"}, start_index=0, end_index=12, content='{123: "John"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.exceptions.ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "invalid_key"
        assert msg.index == [123]
        assert msg.text == "All object keys must be strings."
        assert msg.start_position is not None
        assert msg.end_position is not None


# LLM-generated content at query #9
#--------------------------

def test_validate_with_positions_handles_required_error():
    from typesystem.fields import Field, Schema
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.tokens import Token
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
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


# LLM-generated content at query #10
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    token._get_value = lambda: {"field": None}
    token.lookup = lambda index: Token(value=None, start_index=8, end_index=11, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'field' is required."
    assert message.code == "required"
    assert message.index == ["field"]
    assert message.start_position.char_index == 8
    assert message.end_position.char_index == 11

def test_validate_with_positions_custom_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")
    field = MockField()
    token = Token(value="invalid", start_index=5, end_index=11, content='value: "invalid"')
    token._get_value = lambda: "invalid"
    token.lookup = lambda index: Token(value="invalid", start_index=5, end_index=11, content='value: "invalid"')
    try:
        validate_with_positions(token=token, validator=field)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Custom error."
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position.char_index == 5
    assert message.end_position.char_index == 11

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            raise self.validation_error("type")
    inner_field = MockField()
    inner_schema = Schema(fields={"nested": inner_field})
    outer_schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {"nested": 123}}, start_index=0, end_index=30, content='{"outer": {"nested": 123}}')
    token._get_value = lambda: {"outer": {"nested": 123}}
    token.lookup = lambda index: Token(value=123, start_index=20, end_index=22, content='{"outer": {"nested": 123}}')
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Must be a string."
    assert message.code == "type"
    assert message.index == ["outer", "nested"]
    assert message.start_position.char_index == 20
    assert message.end_position.char_index == 22

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value.upper()
    field = MockField()
    token = Token(value="hello", start_index=0, end_index=4, content='"hello"')
    token._get_value = lambda: "hello"
    result = validate_with_positions(token=token, validator=field)
    assert result == "HELLO"

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid."}
        def validate(self, value):
            raise self.validation_error("invalid")
    field1 = MockField()
    field2 = MockField()
    schema = Schema(fields={"a": field1, "b": field2})
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=20, content='{"a": 1, "b": 2}')
    token._get_value = lambda: {"a": 1, "b": 2}
    token.lookup = lambda index: Token(value=1 if index[-1]=="a" else 2, start_index=7 if index[-1]=="a" else 16, end_index=7 if index[-1]=="a" else 16, content='{"a": 1, "b": 2}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 7
    assert messages[1].start_position.char_index == 16


# LLM-generated content at query #11
#--------------------------

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    token = Token(value="valid", start_index=0, end_index=4, content="valid")
    validator = MockField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "valid"

def test_validate_with_positions_validation_error_without_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Invalid value", code="invalid")
    token = Token(value="invalid", start_index=0, end_index=6, content="invalid")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid value"
        assert msg.code == "invalid"
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 6

def test_validate_with_positions_validation_error_with_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        pass
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=1, content="{}")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'name' is required."
        assert msg.code == "required"
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 1

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Field error", code="custom")
    fields = {"field1": MockField(), "field2": MockField()}
    schema = Schema(fields=fields)
    content = '{"field2": "val", "field1": "val"}'
    token = Token(value={"field2": "val", "field1": "val"}, start_index=0, end_index=len(content)-1, content=content)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index < messages[1].start_position.char_index


# LLM-generated content at query #12
#--------------------------

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    token = Token(value=42, start_index=0, end_index=1, content="42")
    validator = MockField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == 42

def test_validate_with_positions_validation_error_without_index():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Invalid value", code="invalid")
    token = Token(value=42, start_index=0, end_index=1, content="42")
    validator = MockField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid value"
        assert msg.code == "invalid"
        assert msg.index == []
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 2
        assert msg.end_position.char_index == 1

def test_validate_with_positions_validation_error_with_index():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field": 42}
        def _get_child_token(self, key):
            if key == "field":
                return Token(value=42, start_index=8, end_index=9, content='{"field": 42}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Invalid field", code="invalid", index=["field"])
    token = MockToken(value=None, start_index=0, end_index=13, content='{"field": 42}')
    validator = Schema(fields={"field": MockField()})
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Invalid field"
        assert msg.code == "invalid"
        assert msg.index == ["field"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 9
        assert msg.start_position.char_index == 8
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 10
        assert msg.end_position.char_index == 9

def test_validate_with_positions_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
        def lookup(self, index):
            if index == []:
                return self
            raise KeyError
    token = MockToken(value=None, start_index=0, end_index=1, content="{}")
    validator = Schema(fields={"field": Field()})
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'field' is required."
        assert msg.code == "required"
        assert msg.index == ["field"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 1
        assert msg.start_position.char_index == 0
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 2
        assert msg.end_position.char_index == 1

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"a": 1, "b": 2}
        def _get_child_token(self, key):
            if key == "a":
                return Token(value=1, start_index=5, end_index=5, content='{"a":1,"b":2}')
            if key == "b":
                return Token(value=2, start_index=11, end_index=11, content='{"a":1,"b":2}')
            raise KeyError
        def _get_key_token(self, key):
            raise NotImplementedError
        def lookup(self, index):
            if index == []:
                return self
            if len(index) == 1:
                return self._get_child_token(index[0])
            raise KeyError
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Error", code="error", index=[])
    token = MockToken(value=None, start_index=0, end_index=14, content='{"a":1,"b":2}')
    validator = Schema(fields={"a": MockField(), "b": MockField()})
    try:
        validate_with_positions(token=token, validator=validator)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        msg1 = messages[0]
        msg2 = messages[1]
        assert msg1.start_position.char_index == 5
        assert msg2.start_position.char_index == 11


# LLM-generated content at query #13
#--------------------------

def test_validate_with_positions_simple_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("custom")
    token = Token(value="test", start_index=0, end_index=3, content="test")
    field = MockField()
    try:
        validate_with_positions(token=token, validator=field)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Invalid value."
    assert message.code == "custom"
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1
    assert message.start_position.char_index == 0
    assert message.end_position.line_no == 1
    assert message.end_position.column_no == 4
    assert message.end_position.char_index == 3

def test_validate_with_positions_schema_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"existing": "value"}
        def lookup(self, index):
            if index == ["missing"]:
                return Token(value=None, start_index=10, end_index=15, content='{"missing": null}')
            return self
    token = MockToken(value=None, start_index=0, end_index=20, content='{"existing": "value"}')
    schema = Schema(fields={"missing": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'missing' is required."
    assert message.code == "required"
    assert message.index == ["missing"]
    assert message.start_position.char_index == 10
    assert message.end_position.char_index == 15

def test_validate_with_positions_nested_schema_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"nested": {"field": "invalid"}}
        def lookup(self, index):
            if index == ["nested", "field"]:
                return Token(value="invalid", start_index=15, end_index=21, content='{"nested": {"field": "invalid"}}')
            elif index == ["nested"]:
                return Token(value={"field": "invalid"}, start_index=10, end_index=30, content='{"nested": {"field": "invalid"}}')
            return self
    class MockField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
    token = MockToken(value=None, start_index=0, end_index=35, content='{"nested": {"field": "invalid"}}')
    schema = Schema(fields={"nested": Schema(fields={"field": MockField()})})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Must be an integer."
    assert message.code == "type"
    assert message.index == ["nested", "field"]
    assert message.start_position.char_index == 15
    assert message.end_position.char_index == 21

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"a": "invalid1", "b": "invalid2"}
        def lookup(self, index):
            if index == ["a"]:
                return Token(value="invalid1", start_index=5, end_index=12, content='{"a": "invalid1", "b": "invalid2"}')
            if index == ["b"]:
                return Token(value="invalid2", start_index=20, end_index=27, content='{"a": "invalid1", "b": "invalid2"}')
            return self
    class MockField(Field):
        errors = {"custom": "Invalid."}
        def validate(self, value):
            raise self.validation_error("custom")
    token = MockToken(value=None, start_index=0, end_index=30, content='{"a": "invalid1", "b": "invalid2"}')
    schema = Schema(fields={"a": MockField(), "b": MockField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 20

def test_validate_with_positions_successful_validation():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value.upper()
    token = Token(value="test", start_index=0, end_index=3, content="test")
    field = MockField()
    result = validate_with_positions(token=token, validator=field)
    assert result == "TEST"

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    token = Token(value=None, start_index=0, end_index=3, content="null")
    schema = Schema(fields={}, allow_null=True)
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_invalid_key_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {123: "value"}
        def lookup(self, index):
            if index == [123]:
                return Token(value="value", start_index=5, end_index=9, content='{123: "value"}')
            return self
    token = MockToken(value=None, start_index=0, end_index=12, content='{123: "value"}')
    schema = Schema(fields={"valid": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "invalid_key"
    assert message.index == [123]
    assert message.start_position.char_index == 5
    assert message.end_position.char_index == 9


# LLM-generated content at query #14
#--------------------------

def test_validate_with_positions_required_field_error():
    from typesystem.base import Message
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken(value={}, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=key, start_index=0, end_index=0, content="")
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required", index=["field"])
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


# LLM-generated content at query #15
#--------------------------

def test_validate_with_positions_required_error():
    from typesystem.base import Message, ValidationError
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return MockToken({}, 0, 0)
        def _get_key_token(self, key):
            return MockToken(key, 0, 0)
        def _get_position(self, index):
            from typesystem.base import Position
            return Position(1, 1, index)
    field = Field()
    schema = Schema(fields={"field": field})
    token = MockToken({})
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


# LLM-generated content at query #16
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message, ValidationError
    class MockField(Field):
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={}, start_index=0, end_index=1, content="{}")
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
        assert msg.end_position.column_no == 2
        assert msg.end_position.char_index == 1

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message, ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": 123}, start_index=0, end_index=10, content='{"name": 123}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Must be a string."
        assert msg.code == "type"
        assert msg.index == ["name"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 9
        assert msg.start_position.char_index == 8
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 12
        assert msg.end_position.char_index == 11

def test_validate_with_positions_nested_required():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message, ValidationError
    class MockField(Field):
        def validate(self, value):
            return value
    inner_schema = Schema(fields={"inner_name": MockField()})
    fields = {"outer": inner_schema}
    schema = Schema(fields=fields)
    token = Token(value={"outer": {}}, start_index=0, end_index=15, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'inner_name' is required."
        assert msg.code == "required"
        assert msg.index == ["outer", "inner_name"]
        assert msg.start_position.line_no == 1
        assert msg.start_position.column_no == 11
        assert msg.start_position.char_index == 10
        assert msg.end_position.line_no == 1
        assert msg.end_position.column_no == 12
        assert msg.end_position.char_index == 11

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message, ValidationError
    class MockField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=15, content='{"a": 1, "b": 2}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        msg_a = messages[0]
        msg_b = messages[1]
        assert msg_a.index == ["a"]
        assert msg_b.index == ["b"]
        assert msg_a.start_position.char_index == 6
        assert msg_b.start_position.char_index == 13
        assert msg_a.start_position.char_index < msg_b.start_position.char_index

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message, ValidationError
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
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=15, content="")
    token = MockToken(value={}, start_index=0, end_index=30, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
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

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "not an object"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="not an object", start_index=0, end_index=15, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "type"
    assert message.index == []
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 1

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
            self._mock_children = {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self._mock_children.get(key, MockToken(value=None, start_index=10, end_index=20, content=""))
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=15, content="")
    inner_token = MockToken(value={}, start_index=10, end_index=20, content="")
    outer_token = MockToken(value={"inner": {}}, start_index=0, end_index=30, content="")
    outer_token._mock_children["inner"] = inner_token
    inner_field = Field()
    outer_fields = {"inner": Schema(fields={"required_field": Field()})}
    outer_schema = Schema(fields=outer_fields)
    try:
        validate_with_positions(token=outer_token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["inner", "required_field"]
    assert message.start_position.line_no == 1
    assert message.start_position.column_no == 11

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"name": "test"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value={"name": "test"}, start_index=0, end_index=30, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="invalid", start_index=0, end_index=7, content="")
    union = Union(any_of=[Field(), Field()])
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

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
            self._mock_children = {}
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            return self._mock_children.get(key, MockToken(value=None, start_index=0, end_index=0, content=""))
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token1 = MockToken(value={}, start_index=20, end_index=30, content="")
    token2 = MockToken(value={}, start_index=0, end_index=10, content="")
    outer_token = MockToken(value={"field1": {}, "field2": {}}, start_index=0, end_index=40, content="")
    outer_token._mock_children["field1"] = token1
    outer_token._mock_children["field2"] = token2
    fields = {"field1": Schema(fields={"req1": Field()}), "field2": Schema(fields={"req2": Field()})}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=outer_token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index == 0
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #2
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
            return MockToken(value=None, start_index=5, end_index=10, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="")
    token = MockToken(value={}, start_index=0, end_index=4, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "required"
        assert message.index == ["name"]
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "not an integer"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="not an integer", start_index=0, end_index=15, content="")
    class IntegerField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    field = IntegerField()
    try:
        validate_with_positions(token=token, validator=field)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "type"
        assert message.index == []
        assert message.start_position is not None
        assert message.end_position is not None

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
                return MockToken(value={"age": "not an int"}, start_index=5, end_index=30, content="")
            elif key == "age":
                return MockToken(value="not an int", start_index=10, end_index=20, content="")
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value={"user": {"age": "not an int"}}, start_index=0, end_index=35, content="")
    class IntegerField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    fields = {"user": Schema(fields={"age": IntegerField()})}
    schema = Schema(fields=fields)
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "type"
        assert message.index == ["user", "age"]
        assert message.start_position is not None
        assert message.end_position is not None

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"name": "test"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value={"name": "test"}, start_index=0, end_index=20, content="")
    fields = {"name": Field()}
    schema = Schema(fields=fields)
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "test"}

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content="")
    token = MockToken(value="invalid", start_index=0, end_index=7, content="")
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
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "union"
        assert message.index == []
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #3
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
    class MockToken(Token):
        def _get_value(self):
            return {}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value={}, start_index=0, end_index=5, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.index == ["name"]
        assert msg.text == "The field 'name' is required."
        assert msg.start_position is not None
        assert msg.end_position is not None

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
    class MockToken(Token):
        def _get_value(self):
            return {1: "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value={1: "value"}, start_index=0, end_index=5, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "invalid_key"
        assert msg.index == [1]
        assert msg.text == "All object keys must be strings."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_schema_field_validation_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def _get_value(self):
            return {"name": "bad"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value={"name": "bad"}, start_index=0, end_index=5, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "custom"
        assert msg.index == ["name"]
        assert msg.text == "Invalid value."
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        def validate(self, value):
            raise self.validation_error("type")
    class MockField2(Field):
        def validate(self, value):
            raise self.validation_error("type")
    union = Union(any_of=[MockField1(), MockField2()])
    class MockToken(Token):
        def _get_value(self):
            return "value"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value="value", start_index=0, end_index=5, content="")
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "union"
        assert msg.text == "Did not match any valid type."
        assert msg.start_position is not None
        assert msg.end_position is not None

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
    class MockToken(Token):
        def _get_value(self):
            return {"name": "test"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value={"name": "test"}, start_index=0, end_index=5, content="")
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": "TEST"}

def test_validate_with_positions_sorted_messages():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"custom": "Error."}
        def validate(self, value):
            raise self.validation_error("custom")
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    class MockToken(Token):
        def _get_value(self):
            return {"a": 1, "b": 2}
        def _get_child_token(self, key):
            if key == "a":
                return MockToken(value=None, start_index=30, end_index=40, content="")
            else:
                return MockToken(value=None, start_index=10, end_index=20, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content="")
    token = MockToken(value={"a": 1, "b": 2}, start_index=0, end_index=5, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].index == ["b"]
        assert messages[1].index == ["a"]
        assert messages[0].start_position.char_index < messages[1].start_position.char_index


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    import typesystem

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value

    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": None}, start_index=0, end_index=10, content='{"field": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except typesystem.ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field"]
        assert message.text == "The field 'field' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #5
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
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
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.index == ["name"]
    assert message.text == "The field 'name' is required."

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
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "type"
    assert message.index == ["name"]
    assert message.text == "Must be a string."

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"max_length": "Must have no more than 5 characters."}
        def validate(self, value):
            if len(value) > 5:
                raise self.validation_error("max_length")
            return value
    nested_schema = Schema(fields={"title": MockField()})
    fields = {"item": nested_schema}
    schema = Schema(fields=fields)
    token = Token(value={"item": {"title": "too long"}}, start_index=0, end_index=30, content='{"item": {"title": "too long"}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "max_length"
    assert message.index == ["item", "title"]
    assert message.text == "Must have no more than 5 characters."

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"type": "Must be a string.", "max_length": "Too long."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            if len(value) > 5:
                raise self.validation_error("max_length")
            return value
    fields = {"a": MockField(), "b": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"a": 123, "b": "longvalue"}, start_index=0, end_index=30, content='{"a": 123, "b": "longvalue"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0].index == ["a"]
    assert messages[0].code == "type"
    assert messages[1].index == ["b"]
    assert messages[1].code == "max_length"

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

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field, Union
    from typesystem.tokenize.positional_validation import validate_with_positions
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    class StrField(Field):
        errors = {"type": "Must be a string."}
        def validate(self, value):
            if not isinstance(value, str):
                raise self.validation_error("type")
            return value
    union_field = IntField() | StrField()
    fields = {"data": union_field}
    schema = Schema(fields=fields)
    token = Token(value={"data": 3.14}, start_index=0, end_index=15, content='{"data": 3.14}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "union"
    assert message.index == ["data"]
    assert message.text == "Did not match any valid type."

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def __init__(self, **kwargs):
            kwargs["allow_null"] = True
            super().__init__(**kwargs)
        def validate(self, value):
            return value
    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    token = Token(value={"name": None}, start_index=0, end_index=15, content='{"name": null}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"name": None}


# LLM-generated content at query #6
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
            return MockToken(value=None, start_index=5, end_index=10, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={}, start_index=0, end_index=4, content="content")
    schema = Schema(fields={"field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["field"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_nested_required_field():
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
                return MockToken(value={}, start_index=7, end_index=12, content="content")
            return MockToken(value=None, start_index=14, end_index=19, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={"outer": {}}, start_index=0, end_index=20, content="content")
    inner_schema = Schema(fields={"inner": Field()})
    schema = Schema(fields={"outer": inner_schema})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.index == ["outer", "inner"]
        assert msg.start_position is not None
        assert msg.end_position is not None

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
    class MockToken(Token):
        def _get_value(self):
            return {"field": "invalid"}
        def _get_child_token(self, key):
            return MockToken(value="invalid", start_index=8, end_index=15, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={"field": "invalid"}, start_index=0, end_index=20, content="content")
    schema = Schema(fields={"field": MockField()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid"
        assert msg.index == ["field"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_key_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {123: "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={123: "value"}, start_index=0, end_index=20, content="content")
    schema = Schema(fields={"field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "invalid_key"
        assert msg.index == [123]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content="", child_start=5, child_end=10):
            super().__init__(value, start_index, end_index, content)
            self.child_start = child_start
            self.child_end = child_end
        def _get_value(self):
            return self._value
        def _get_child_token(self, key):
            if key == "a":
                return MockToken(value=None, start_index=1, end_index=2, content="content")
            return MockToken(value=None, start_index=3, end_index=4, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={"a": None, "b": None}, start_index=0, end_index=20, content="content")
    schema = Schema(fields={"a": Field(), "b": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field": "valid"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value={"field": "valid"}, start_index=0, end_index=20, content="content")
    schema = Schema(fields={"field": Field()})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "valid"}

def test_validate_with_positions_null_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return None
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value=None, start_index=0, end_index=4, content="content")
    schema = Schema(fields={}, allow_null=True)
    result = validate_with_positions(token=token, validator=schema)
    assert result is None

def test_validate_with_positions_null_not_allowed():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return None
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=10, content="content")
    token = MockToken(value=None, start_index=0, end_index=4, content="


# LLM-generated content at query #7
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return MockToken(self._value[key], start_index=1, end_index=2, content="ab")

        def _get_key_token(self, key):
            return MockToken(key, start_index=0, end_index=0, content="a")

        @property
        def start(self):
            return self._get_position(self._start_index)

        @property
        def end(self):
            return self._get_position(self._end_index)

        def _get_position(self, index):
            from typesystem.tokenize.tokens import Position
            return Position(line_no=1, column_no=index + 1, char_index=index)

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")

    schema = Schema(fields={"field": MockField()})
    token = MockToken({})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'field' is required."
        assert msg.code == "required"
        assert msg.index == ["field"]
        assert msg.start_position.char_index == 1
        assert msg.end_position.char_index == 2

def test_validate_with_positions_custom_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return MockToken(self._value[key], start_index=1, end_index=2, content="ab")

        def _get_key_token(self, key):
            return MockToken(key, start_index=0, end_index=0, content="a")

        @property
        def start(self):
            return self._get_position(self._start_index)

        @property
        def end(self):
            return self._get_position(self._end_index)

        def _get_position(self, index):
            from typesystem.tokenize.tokens import Position
            return Position(line_no=1, column_no=index + 1, char_index=index)

    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")

    schema = Schema(fields={"field": MockField()})
    token = MockToken({"field": "value"})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Custom error."
        assert msg.code == "custom"
        assert msg.index == ["field"]
        assert msg.start_position.char_index == 1
        assert msg.end_position.char_index == 2

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            if isinstance(key, str):
                return MockToken(self._value[key], start_index=1, end_index=2, content="ab")
            else:
                return MockToken(self._value[key], start_index=3, end_index=4, content="cd")

        def _get_key_token(self, key):
            return MockToken(key, start_index=0, end_index=0, content="a")

        @property
        def start(self):
            return self._get_position(self._start_index)

        @property
        def end(self):
            return self._get_position(self._end_index)

        def _get_position(self, index):
            from typesystem.tokenize.tokens import Position
            return Position(line_no=1, column_no=index + 1, char_index=index)

    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self, value):
            raise self.validation_error("custom")

    inner_schema = Schema(fields={"inner_field": MockField()})
    outer_schema = Schema(fields={"outer_field": inner_schema})
    token = MockToken({"outer_field": {"inner_field": "value"}})
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Custom error."
        assert msg.code == "custom"
        assert msg.index == ["outer_field", "inner_field"]
        assert msg.start_position.char_index == 3
        assert msg.end_position.char_index == 4

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return MockToken(self._value[key], start_index=1, end_index=2, content="ab")

        def _get_key_token(self, key):
            return MockToken(key, start_index=0, end_index=0, content="a")

        @property
        def start(self):
            return self._get_position(self._start_index)

        @property
        def end(self):
            return self._get_position(self._end_index)

        def _get_position(self, index):
            from typesystem.tokenize.tokens import Position
            return Position(line_no=1, column_no=index + 1, char_index=index)

    class MockField(Field):
        def validate(self, value):
            return value

    schema = Schema(fields={"field": MockField()})
    token = MockToken({"field": "value"})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "value"}

def test_validate_with_positions_sorts_messages_by_position():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.exceptions import ValidationError

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._value = value

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            if key == "field1":
                return MockToken(self._value[key], start_index=10, end_index=15, content="content")
            else:
                return MockToken(self._value[key], start_index=5, end_index=8, content="content")

        def _get_key_token(self, key):
            return MockToken(key, start_index=0, end_index=0, content="a")

        @property
        def start(self):
            return self._get_position(self._start_index)

        @property
        def end(self):
            return self._get_position(self._end_index)

        def _get_position(self, index):
            from typesystem.tokenize.tokens import Position
            return Position(line_no=1, column_no=index + 1, char_index=index)

    class MockField(Field):
        errors = {"custom": "Custom error."}
        def validate(self,


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions_handles_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    import typesystem.tokenize.positional_validation as module

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            raise self.validation_error("required")

    class MockToken(Token):
        def _get_value(self):
            return {"existing": "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=10, end_index=20, content=" " * 30)
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=5, end_index=15, content=" " * 30)

    schema = Schema(fields={"missing": MockField()})
    token = MockToken(value={"existing": "value"}, start_index=0, end_index=25, content=" " * 30)
    try:
        module.validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["missing"]
        assert message.text == "The field 'missing' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #9
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
            return MockToken(
                value={},
                start_index=self._start_index,
                end_index=self._end_index,
                content=self._content
            )

        def _get_key_token(self, key):
            return MockToken(
                value=key,
                start_index=self._start_index,
                end_index=self._end_index,
                content=self._content
            )

    token = MockToken(value={}, start_index=0, end_index=10, content=" " * 11)
    schema = Schema(fields={"required_field": Field()})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["required_field"]
        assert message.start_position == token.start
        assert message.end_position == token.end
        assert message.text == "The field 'required_field' is required."


# LLM-generated content at query #10
#--------------------------

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    mock_token = Token(value={"key": "valid"}, start_index=0, end_index=10, content='{"key": "valid"}')
    mock_field = MockField()
    result = validate_with_positions(token=mock_token, validator=mock_field)
    assert result == {"key": "valid"}

def test_validate_with_positions_validation_error_without_positions():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Invalid value", code="invalid")
    mock_token = Token(value="invalid", start_index=0, end_index=6, content='"invalid"')
    mock_field = MockField()
    try:
        validate_with_positions(token=mock_token, validator=mock_field)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Invalid value"
        assert messages[0].code == "invalid"
        assert messages[0].start_position.line_no == 1
        assert messages[0].start_position.column_no == 1
        assert messages[0].end_position.line_no == 1
        assert messages[0].end_position.column_no == 9

def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required", index=["field"])
    mock_token = Token(value={}, start_index=0, end_index=2, content='{}')
    mock_schema = Schema(fields={"field": MockField()})
    try:
        validate_with_positions(token=mock_token, validator=mock_schema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "The field 'field' is required."
        assert messages[0].code == "required"
        assert messages[0].index == ["field"]
        assert messages[0].start_position.line_no == 1
        assert messages[0].start_position.column_no == 1
        assert messages[0].end_position.line_no == 1
        assert messages[0].end_position.column_no == 3

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="Invalid nested", code="nested", index=["outer", "inner"])
    mock_token = Token(value={"outer": {"inner": "bad"}}, start_index=0, end_index=25, content='{"outer": {"inner": "bad"}}')
    mock_schema = Schema(fields={"outer": Schema(fields={"inner": MockField()})})
    try:
        validate_with_positions(token=mock_token, validator=mock_schema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].text == "Invalid nested"
        assert messages[0].code == "nested"
        assert messages[0].index == ["outer", "inner"]
        assert messages[0].start_position.line_no == 1
        assert messages[0].start_position.column_no == 11
        assert messages[0].end_position.line_no == 1
        assert messages[0].end_position.column_no == 25

def test_validate_with_positions_multiple_errors_sorted():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField1(Field):
        def validate(self, value):
            raise ValidationError(text="Error first", code="error1", index=["a"])
    class MockField2(Field):
        def validate(self, value):
            raise ValidationError(text="Error second", code="error2", index=["b"])
    mock_token = Token(value={"a": 1, "b": 2}, start_index=0, end_index=13, content='{"a":1,"b":2}')
    mock_schema = Schema(fields={"a": MockField1(), "b": MockField2()})
    try:
        validate_with_positions(token=mock_token, validator=mock_schema)
        assert False
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].code == "error1"
        assert messages[0].start_position.char_index < messages[1].start_position.char_index
        assert messages[1].code == "error2"


# LLM-generated content at query #11
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
    schema = Schema(fields={"field": field})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'field' is required."
    assert message.code == "required"
    assert message.index == ["field"]

def test_validate_with_positions_nested_required_field():
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
                return MockToken(value={}, start_index=8, end_index=15, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"outer": {}}, start_index=0, end_index=20, content="")
    inner_field = Field()
    inner_schema = Schema(fields={"inner": inner_field})
    outer_schema = Schema(fields={"outer": inner_schema})
    try:
        validate_with_positions(token=token, validator=outer_schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]

def test_validate_with_positions_non_required_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"field": "invalid"}
        def _get_child_token(self, key):
            return MockToken(value="invalid", start_index=8, end_index=15, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value={"field": "invalid"}, start_index=0, end_index=20, content="")
    class CustomField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    field = CustomField()
    schema = Schema(fields={"field": field})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be an integer."
    assert message.code == "type"
    assert message.index == ["field"]

def test_validate_with_positions_sorts_messages_by_position():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def __init__(self, value, start_index, end_index, content=""):
            super().__init__(value, start_index, end_index, content)
            self._position_cache = {}
        def _get_value(self):
            return {"field1": "invalid1", "field2": "invalid2"}
        def _get_child_token(self, key):
            if key == "field1":
                return MockToken(value="invalid1", start_index=10, end_index=18, content="")
            elif key == "field2":
                return MockToken(value="invalid2", start_index=20, end_index=28, content="")
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_position(self, index):
            if index not in self._position_cache:
                self._position_cache[index] = super()._get_position(index)
            return self._position_cache[index]
    token = MockToken(value={"field1": "invalid1", "field2": "invalid2"}, start_index=0, end_index=30, content="")
    class CustomField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    field1 = CustomField()
    field2 = CustomField()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert len(error.messages) == 2
    assert error.messages[0].index == ["field1"]
    assert error.messages[1].index == ["field2"]

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return "invalid"
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=5, content="")
    token = MockToken(value="invalid", start_index=0, end_index=7, content="")
    class IntField(Field):
        errors = {"type": "Must be an integer."}
        def validate(self, value):
            if not isinstance(value, int):
                raise self.validation_error("type")
            return value
    class StrField(Field):
        errors = {"max_length": "Must be at most 5 characters."}
        def validate(self, value):
            if len(value) > 5:
                raise self.validation_error("max_length")
            return value
    union = Union(any_of=[IntField(), StrField()])
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert len(error.messages) == 1
    message = error.messages[0]
    assert message.text == "Must be at most 5 characters."
    assert message.code == "max_length"
    assert message.index == []


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions

    class MockToken(Token):
        def __init__(self, value, start_index=0, end_index=0, content=""):
            super().__init__(value, start_index, end_index, content)
            self._child_tokens = {}

        def _get_value(self):
            return self._value

        def _get_child_token(self, key):
            return self._child_tokens.get(key, MockToken(None, 0, 0))

        def _get_key_token(self, key):
            return MockToken(key, 0, 0)

        def lookup(self, index):
            token = self
            for key in index:
                token = token._get_child_token(key)
            return token

    class MockField(Field):
        def validate(self, value):
            raise ValidationError(text="This field is required.", code="required")

    schema = Schema(fields={"field1": MockField()})
    token = MockToken({})
    token._child_tokens = {"field1": MockToken(None, 10, 20, "test content")}
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'field1' is required."
        assert message.index == ["field1"]
        assert message.start_position.line_no == 1
        assert message.start_position.column_no == 1
        assert message.start_position.char_index == 10
        assert message.end_position.line_no == 1
        assert message.end_position.column_no == 1
        assert message.end_position.char_index == 20


# LLM-generated content at query #13
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
            return MockToken(value=None, start_index=0, end_index=5, content='{"key":')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=2, end_index=4, content='{"key":')
    token = MockToken(value={}, start_index=0, end_index=5, content='{"key":')
    field = Field()
    schema = Schema(fields={"key": field})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "The field 'key' is required."
        assert msg.code == "required"
        assert msg.index == ["key"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_invalid_type():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"key": 123}
        def _get_child_token(self, key):
            if key == "key":
                return MockToken(value=123, start_index=7, end_index=9, content='{"key": 123}')
            return MockToken(value=None, start_index=0, end_index=12, content='{"key": 123}')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=2, end_index=4, content='{"key": 123}')
    token = MockToken(value={"key": 123}, start_index=0, end_index=12, content='{"key": 123}')
    field = Field()
    schema = Schema(fields={"key": field})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Must be an object."
        assert msg.code == "type"
        assert msg.index == []
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_nested_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"nested": {"inner": 456}}
        def _get_child_token(self, key):
            if key == "nested":
                return MockToken(value={"inner": 456}, start_index=10, end_index=30, content='{"nested": {"inner": 456}}')
            if key == "inner":
                return MockToken(value=456, start_index=20, end_index=22, content='{"nested": {"inner": 456}}')
            return MockToken(value=None, start_index=0, end_index=32, content='{"nested": {"inner": 456}}')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=12, end_index=17, content='{"nested": {"inner": 456}}')
    token = MockToken(value={"nested": {"inner": 456}}, start_index=0, end_index=32, content='{"nested": {"inner": 456}}')
    inner_field = Field()
    nested_schema = Schema(fields={"inner": inner_field})
    schema = Schema(fields={"nested": nested_schema})
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Must be an object."
        assert msg.code == "type"
        assert msg.index == ["nested"]
        assert msg.start_position is not None
        assert msg.end_position is not None

def test_validate_with_positions_success():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.fields import Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return {"key": "value"}
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content='')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=0, content='')
    token = MockToken(value={"key": "value"}, start_index=0, end_index=0, content='')
    field = Field()
    schema = Schema(fields={"key": field})
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"key": "value"}

def test_validate_with_positions_union_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Union, Field
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockToken(Token):
        def _get_value(self):
            return 123
        def _get_child_token(self, key):
            return MockToken(value=None, start_index=0, end_index=2, content='123')
        def _get_key_token(self, key):
            return MockToken(value=None, start_index=0, end_index=2, content='123')
    token = MockToken(value=123, start_index=0, end_index=2, content='123')
    union = Union(any_of=[Field(), Field()])
    try:
        validate_with_positions(token=token, validator=union)
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.text == "Did not match any valid type."
        assert msg.code == "union"
        assert msg.index == []
        assert msg.start_position is not None
        assert msg.end_position is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions_required_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError

    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value

    schema = Schema(fields={"field1": MockField()})
    token = Token(value={"field1": None}, start_index=0, end_index=20, content='{"field1": null}')
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.index == ["field1"]
        assert message.text == "The field 'field1' is required."
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #15
#--------------------------

def test_validate_with_positions_required_field_error():
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
            raise ValidationError(text="This field is required.", code="required")
    schema = Schema(fields={"field": MockField()})
    token = MockToken(value={}, start_index=0, end_index=0, content="")
    try:
        validate_with_positions(token=token, validator=schema)
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'field' is required."
        assert message.index == ["field"]
        assert message.start_position is not None
        assert message.end_position is not None


# LLM-generated content at query #16
#--------------------------

def test_validate_with_positions_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"other": "value"}, start_index=0, end_index=10, content='{"other": "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'field' is required."
    assert message.code == "required"
    assert message.index == ["field"]

def test_validate_with_positions_nested_required_field():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"required": "This field is required."}
        def validate(self, value):
            if value is None:
                raise self.validation_error("required")
            return value
    field = MockField()
    inner_schema = Schema(fields={"inner": field})
    schema = Schema(fields={"outer": inner_schema})
    token = Token(value={"outer": {}}, start_index=0, end_index=20, content='{"outer": {}}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "The field 'inner' is required."
    assert message.code == "required"
    assert message.index == ["outer", "inner"]

def test_validate_with_positions_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": "bad"}, start_index=0, end_index=15, content='{"field": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Invalid value."
    assert message.code == "invalid"
    assert message.index == ["field"]

def test_validate_with_positions_multiple_errors():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    field1 = MockField()
    field2 = MockField()
    schema = Schema(fields={"field1": field1, "field2": field2})
    token = Token(value={"field1": "bad", "field2": "bad"}, start_index=0, end_index=30, content='{"field1": "bad", "field2": "bad"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 2
    messages = error.messages()
    assert messages[0].code == "invalid"
    assert messages[0].index == ["field1"]
    assert messages[1].code == "invalid"
    assert messages[1].index == ["field2"]

def test_validate_with_positions_no_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field
    from typesystem.schemas import Schema
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        def validate(self, value):
            return value
    field = MockField()
    schema = Schema(fields={"field": field})
    token = Token(value={"field": "good"}, start_index=0, end_index=17, content='{"field": "good"}')
    result = validate_with_positions(token=token, validator=schema)
    assert result == {"field": "good"}

def test_validate_with_positions_union_field_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.fields import Field, Union
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    class MockField(Field):
        errors = {"invalid": "Invalid value."}
        def validate(self, value):
            raise self.validation_error("invalid")
    field1 = MockField()
    field2 = MockField()
    union = Union(any_of=[field1, field2])
    token = Token(value="bad", start_index=0, end_index=3, content='"bad"')
    try:
        validate_with_positions(token=token, validator=union)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Invalid value."
    assert message.code == "invalid"
    assert message.index == []

def test_validate_with_positions_schema_type_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    schema = Schema(fields={})
    token = Token(value="not an object", start_index=0, end_index=13, content='"not an object"')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "Must be an object."
    assert message.code == "type"
    assert message.index == []

def test_validate_with_positions_schema_null_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    schema = Schema(fields={}, allow_null=False)
    token = Token(value=None, start_index=0, end_index=4, content="null")
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "May not be null."
    assert message.code == "null"
    assert message.index == []

def test_validate_with_positions_schema_invalid_key_error():
    from typesystem.tokenize.tokens import Token
    from typesystem.schemas import Schema
    from typesystem.base import Message
    from typesystem.tokenize.positional_validation import validate_with_positions
    schema = Schema(fields={})
    token = Token(value={1: "value"}, start_index=0, end_index=12, content='{1: "value"}')
    try:
        validate_with_positions(token=token, validator=schema)
    except Exception as e:
        error = e
    assert isinstance(error, ValidationError)
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.text == "All object keys must be strings."
    assert message.code == "invalid_key"
    assert message.index == [1]


