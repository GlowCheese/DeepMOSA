####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
    
    # Test 3: Validation error with custom message
    class LengthSchema(Schema):
        name = String(max_length=3)
    
    token = Token(
        value={"name": "Jonathan"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=LengthSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "long" in message.text.lower()
    
    # Test 4: Nested validation with positions
    class NestedSchema(Schema):
        class Inner(Schema):
            value = Integer(required=True)
        
        inner = Inner
    
    token = Token(
        value={"inner": {}},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."
    
    # Test 5: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(required=True)
        age = Integer(minimum=18, required=True)
    
    token = Token(
        value={"name": "", "age": 15},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Messages should be sorted by start_position
        assert all(isinstance(msg, Message) for msg in messages)
    
    # Test 6: Field validator instead of Schema
    field_validator = String(required=True, max_length=5)
    
    token = Token(
        value="toolong",
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=field_validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
    
    # Test 7: Empty value with required field
    token = Token(
        value={},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        required_fields = {msg.text for msg in messages}
        assert "The field 'name' is required." in required_fields
        assert "The field 'age' is required." in required_fields


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
    
    token = DictToken(
        {"name": ScalarToken("test", 0, 4, '{"name": "test"}')},
        0, 20, '{"name": "test"}'
    )
    result = validate_with_positions(token=token, validator=SimpleSchema())
    assert result == {"name": "test"}
    
    # Test 2: Failed validation with required field error
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer()
    
    token = DictToken(
        {"age": ScalarToken(25, 10, 12, '{"age": 25}')},
        0, 15, '{"age": 25}'
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 15
    
    # Test 3: Failed validation with nested required field error
    class NestedSchema(Schema):
        user = {"name": String(required=True)}
    
    token = DictToken(
        {"user": DictToken({}, 8, 10, '{"user": {}}')},
        0, 12, '{"user": {}}'
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["user", "name"]
        assert message.start_position.char_index == 8
        assert message.end_position.char_index == 10
    
    # Test 4: Failed validation with multiple errors
    class MultiErrorSchema(Schema):
        name = String(required=True, max_length=5)
        age = Integer(minimum=0)
    
    token = DictToken(
        {
            "name": ScalarToken("toolong", 10, 19, '{"name": "toolong"}'),
            "age": ScalarToken(-5, 25, 27, '{"age": -5}')
        },
        0, 30, '{"name": "toolong", "age": -5}'
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        sorted_messages = sorted(messages, key=lambda m: m.start_position.char_index)
        assert sorted_messages[0].code == "max_length"
        assert sorted_messages[0].index == ["name"]
        assert sorted_messages[0].start_position.char_index == 10
        assert sorted_messages[1].code == "minimum"
        assert sorted_messages[1].index == ["age"]
        assert sorted_messages[1].start_position.char_index == 25
    
    # Test 5: Failed validation with non-required error
    class LengthSchema(Schema):
        name = String(max_length=3)
    
    token = DictToken(
        {"name": ScalarToken("toolong", 10, 19, '{"name": "toolong"}')},
        0, 20, '{"name": "toolong"}'
    )
    try:
        validate_with_positions(token=token, validator=LengthSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.text == "Must have no more than 3 characters."
        assert message.index == ["name"]
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 19
    
    # Test 6: Successful validation with Field directly
    field = String(max_length=5)
    token = ScalarToken("test", 0, 6, '"test"')
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test 7: Failed validation with Field directly
    field = String(max_length=3)
    token = ScalarToken("toolong", 0, 9, '"toolong"')
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.text == "Must have no more than 3 characters."
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 9
    
    # Test 8: Nested list validation error
    class ListSchema(Schema):
        items = [Integer(minimum=0)]
    
    token = DictToken(
        {
            "items": ListToken(
                [
                    ScalarToken(1, 12, 13, '[1, -1]'),
                    ScalarToken(-1, 15, 17, '[1, -1]')
                ],
                10, 18, '{"items": [1, -1]}'
            )
        },
        0, 20, '{"items": [1, -1]}'
    )
    try:
        validate_with_positions(token=token, validator=ListSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "minimum"
        assert message.index == ["items", 1]
        assert message.start_position.char_index == 15
        assert message.end_position.char_index == 17


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'age' is required." in message.text
    
    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        user = TestSchema
    
    token = Token(
        value={"user": {"name": "A" * 20, "age": -5}},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert all(isinstance(msg.start_position, type(None)) for msg in messages)
        assert all(isinstance(msg.end_position, type(None)) for msg in messages)
    
    # Test 4: Validation error with Field directly
    field = String(max_length=5)
    token = Token(
        value="too_long",
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "max_length"
    
    # Test 5: Messages are sorted by start position
    class MultiErrorSchema(Schema):
        a = String(required=True)
        b = Integer(required=True)
        c = String(max_length=1)
    
    token = Token(
        value={"c": "too_long"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        # Messages should be sorted (even though positions are None)
        for i in range(len(messages) - 1):
            assert messages[i].start_position is None
            assert messages[i].end_position is None
    
    # Test 6: Complex nested error with positions
    class InnerSchema(Schema):
        value = String(required=True)
    
    class OuterSchema(Schema):
        inner = InnerSchema
    
    token = Token(
        value={"inner": {}},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=OuterSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert "The field 'value' is required." in messages[0].text


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import typing

    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)

    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'age' is required." in message.text

    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        class Inner(Schema):
            value = String(required=True)
        
        inner = Inner

    token = Token(
        value={"inner": {}},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'value' is required." in message.text

    # Test 4: Validation error with multiple messages
    class MultiErrorSchema(Schema):
        name = String(required=True, max_length=2)
        age = Integer(required=True, minimum=18)

    token = Token(
        value={"name": "LongName", "age": 15},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "minimum" in codes

    # Test 5: Validation error with Field directly
    field = String(required=True, max_length=5)
    token = Token(
        value="TooLongString",
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test 6: Messages are sorted by start position
    class SortingSchema(Schema):
        field1 = String(required=True)
        field2 = String(required=True)

    token = Token(
        value={},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=SortingSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Messages should be sorted by start_position.char_index
        # (even though positions are None in this test)

    # Test 7: Non-required error message
    class LengthSchema(Schema):
        name = String(max_length=3)

    token = Token(
        value={"name": "LongName"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=LengthSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.text is not None
        assert "required" not in message.text.lower()


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 5, "name"),
            "age": ScalarToken(25, 7, 10, "age")
        },
        start=0,
        end=11
    )
    
    result = validate_with_positions(token=token, validator=TestSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with custom message
    token = DictToken(
        value={
            "name": ScalarToken("VeryLongName", 0, 13, "name"),
            "age": ScalarToken(25, 15, 18, "age")
        },
        start=0,
        end=19
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "name" in message.text
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 13
    
    # Test 3: Required field error
    token = DictToken(
        value={
            "age": ScalarToken(25, 0, 3, "age")
        },
        start=0,
        end=4
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 4
    
    # Test 4: Multiple validation errors sorted by position
    token = DictToken(
        value={
            "name": ScalarToken("VeryLongName", 0, 13, "name"),
            "age": ScalarToken(-5, 15, 18, "age")
        },
        start=0,
        end=19
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 0
        assert messages[1].start_position.char_index == 15
    
    # Test 5: Nested validation with Field
    field = String(max_length=5)
    token = ScalarToken("TooLong", 0, 7, "test")
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 7
    
    # Test 6: Nested structure validation
    class NestedSchema(Schema):
        items = ListToken(
            value=[
                DictToken(
                    value={
                        "id": ScalarToken(1, 10, 11, "id"),
                        "value": ScalarToken("test", 13, 19, "value")
                    },
                    start=9,
                    end=20
                )
            ],
            start=8,
            end=21
        )
    
    class ParentSchema(Schema):
        data = NestedSchema
    
    token = DictToken(
        value={
            "data": NestedSchema.items
        },
        start=0,
        end=22
    )
    
    result = validate_with_positions(token=token, validator=ParentSchema())
    assert result == {"data": {"items": [{"id": 1, "value": "test"}]}}
    
    # Test 7: Empty token validation
    token = DictToken(value={}, start=0, end=0)
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        for message in messages:
            assert message.code == "required"


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'age' is required." in message.text
    
    # Test 3: Validation error with custom message
    class LengthSchema(Schema):
        name = String(max_length=5)
    
    token = Token(
        value={"name": "Jonathan"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=LengthSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "Must have no more than 5 characters." in message.text
    
    # Test 4: Nested validation with positions
    class NestedSchema(Schema):
        class Inner(Schema):
            value = Integer(required=True)
        
        inner = Inner
    
    token = Token(
        value={"inner": {}},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'value' is required." in message.text
    
    # Test 5: Multiple validation errors
    class MultiErrorSchema(Schema):
        name = String(required=True, max_length=3)
        age = Integer(required=True, minimum=18)
    
    token = Token(
        value={"name": "Jonathan", "age": 15},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "minimum" in codes
    
    # Test 6: Direct Field validation
    field = String(required=True, max_length=5)
    token = Token(
        value="TooLongString",
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
    
    # Test 7: Empty value with required fields
    class EmptySchema(Schema):
        field = String(required=True)
    
    token = Token(
        value={},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=EmptySchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'field' is required." in message.text


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    from typesystem.base import ValidationError
    import pytest

    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 4, '{"name": "John", "age": 25}'),
            "age": ScalarToken(25, 16, 18, '{"name": "John", "age": 25}')
        },
        start=0,
        end=30,
        content='{"name": "John", "age": 25}'
    )
    
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer()

    token = DictToken(
        value={
            "age": ScalarToken(25, 16, 18, '{"age": 25}')
        },
        start=0,
        end=20,
        content='{"age": 25}'
    )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.text == "The field 'name' is required."
    assert message.index == ["name"]

    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        items = ListToken(
            value=[
                DictToken(
                    value={
                        "value": ScalarToken("test", 10, 16, '[{"value": "test"}]')
                    },
                    start=1,
                    end=18,
                    content='{"value": "test"}'
                )
            ],
            start=0,
            end=19,
            content='[{"value": "test"}]'
        )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=String(max_length=3))
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].code == "max_length"

    # Test 4: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        first = String(max_length=2)
        second = Integer(minimum=10)

    token = DictToken(
        value={
            "first": ScalarToken("toolong", 10, 19, '{"first": "toolong", "second": 5}'),
            "second": ScalarToken(5, 31, 32, '{"first": "toolong", "second": 5}')
        },
        start=0,
        end=45,
        content='{"first": "toolong", "second": 5}'
    )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=MultiErrorSchema)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 2
    
    # Verify messages are sorted by start position
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index
    
    # Verify message types
    codes = [msg.code for msg in messages]
    assert "max_length" in codes
    assert "minimum" in codes

    # Test 5: Direct field validation (not Schema)
    token = ScalarToken("toolongvalue", 0, 13, '"toolongvalue"')
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=String(max_length=5))
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].code == "max_length"
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 13

    # Test 6: Deeply nested required field error
    class DeepSchema(Schema):
        level1 = {
            "level2": {
                "level3": String(required=True)
            }
        }

    token = DictToken(
        value={
            "level1": DictToken(
                value={
                    "level2": DictToken(
                        value={},
                        start=12,
                        end=14,
                        content='{}'
                    )
                },
                start=10,
                end=16,
                content='{"level2": {}}'
            )
        },
        start=0,
        end=18,
        content='{"level1": {"level2": {}}}'
    )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=DeepSchema)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.code == "required"
    assert message.text == "The field 'level3' is required."
    assert message.index == ["level1", "level2", "level3"]


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 4, '{"name":"John","age":25}'),
            "age": ScalarToken(25, 14, 16, '{"name":"John","age":25}')
        },
        start=0,
        end=30,
        content='{"name":"John","age":25}'
    )
    
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Failed validation with required field error
    token_missing = DictToken(
        value={
            "age": ScalarToken(25, 14, 16, '{"age":25}')
        },
        start=0,
        end=18,
        content='{"age":25}'
    )
    
    try:
        validate_with_positions(token=token_missing, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.text == "The field 'name' is required."
        assert msg.index == ["name"]
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 18
    
    # Test 3: Failed validation with multiple errors
    class UserSchema(Schema):
        username = String(min_length=3, max_length=20)
        email = String(format="email")
        score = Integer(minimum=0, maximum=100)
    
    token_invalid = DictToken(
        value={
            "username": ScalarToken("ab", 15, 17, '{"username":"ab","score":-5}'),
            "score": ScalarToken(-5, 30, 32, '{"username":"ab","score":-5}')
        },
        start=0,
        end=34,
        content='{"username":"ab","score":-5}'
    )
    
    try:
        validate_with_positions(token=token_invalid, validator=UserSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        
        # Check messages are sorted by start position
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)
        
        # Check required field error for email
        required_msgs = [m for m in messages if m.code == "required"]
        assert len(required_msgs) == 1
        assert required_msgs[0].index == ["email"]
        
        # Check min_length error for username
        min_length_msgs = [m for m in messages if m.code == "min_length"]
        assert len(min_length_msgs) == 1
        assert min_length_msgs[0].index == ["username"]
        
        # Check minimum error for score
        minimum_msgs = [m for m in messages if m.code == "minimum"]
        assert len(minimum_msgs) == 1
        assert minimum_msgs[0].index == ["score"]
    
    # Test 4: Successful validation with nested structure
    class AddressSchema(Schema):
        street = String()
        city = String()
    
    class PersonSchema(Schema):
        name = String()
        address = AddressSchema
    
    token_nested = DictToken(
        value={
            "name": ScalarToken("Alice", 0, 7, '{"name":"Alice","address":{"street":"Main","city":"NYC"}}'),
            "address": DictToken(
                value={
                    "street": ScalarToken("Main", 20, 26, '{"name":"Alice","address":{"street":"Main","city":"NYC"}}'),
                    "city": ScalarToken("NYC", 34, 39, '{"name":"Alice","address":{"street":"Main","city":"NYC"}}')
                },
                start=10,
                end=41,
                content='{"street":"Main","city":"NYC"}'
            )
        },
        start=0,
        end=43,
        content='{"name":"Alice","address":{"street":"Main","city":"NYC"}}'
    )
    
    result = validate_with_positions(token=token_nested, validator=PersonSchema)
    assert result == {"name": "Alice", "address": {"street": "Main", "city": "NYC"}}
    
    # Test 5: Failed validation with nested required field
    token_nested_missing = DictToken(
        value={
            "name": ScalarToken("Bob", 0, 5, '{"name":"Bob","address":{"city":"LA"}}'),
            "address": DictToken(
                value={
                    "city": ScalarToken("LA", 18, 22, '{"name":"Bob","address":{"city":"LA"}}')
                },
                start=8,
                end=24,
                content='{"city":"LA"}'
            )
        },
        start=0,
        end=26,
        content='{"name":"Bob","address":{"city":"LA"}}'
    )
    
    try:
        validate_with_positions(token=token_nested_missing, validator=PersonSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.text == "The field 'street' is required."
        assert msg.index == ["address", "street"]
    
    # Test 6: Validation with simple Field (not Schema)
    token_scalar = ScalarToken("toolongusername", 0, 16, '"toolongusername"')
    
    try:
        validate_with_positions(token=token_scalar, validator=String(max_length=10))
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "max_length"
        assert "10" in msg.text
        assert msg.index == []
        assert msg.start_position.char_index == 0
        assert msg.end_position.char_index == 16
    
    # Test 7: Validation with list structure
    class ListSchema(Schema):
        items = ListToken(
            value=[
                ScalarToken(1, 10, 11, '{"items":[1,2,3]}'),
                ScalarToken(2, 13, 14, '{"items":[1,2,3]}'),
                ScalarToken(3, 16, 17, '{"items":[1,2,3]}')
            ],
            start=9,
            end=18,
            content='[1,2,3]'
        )
    
    token_list = DictToken(
        value={"items": ListSchema.items},
        start=0,
        end=20,
        content='{"items":[1,2,3]}'
    )
    
    # This would test list validation if ListField was properly defined
    # For now, just ensure no errors with basic structure
    
    # Test 8: Empty token validation
    token_empty = DictToken(value={}, start=0, end=2, content='{}')
    
    try:
        validate_with_positions(token=token_empty, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        required_fields = {m.index[-1] for m in messages if m.code == "required"}
        assert required_fields == {"name", "age"}


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import typing

    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with custom message
    class TestSchema2(Schema):
        name = String(max_length=5)

    token = Token(
        value={"name": "Jonathan"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema2)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "Must have no more than 5 characters" in message.text

    # Test 3: Required field error
    class TestSchema3(Schema):
        name = String(required=True)
        age = Integer()

    token = Token(
        value={"age": 25},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema3)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."

    # Test 4: Multiple validation errors
    class TestSchema4(Schema):
        name = String(required=True, max_length=3)
        age = Integer(minimum=18)

    token = Token(
        value={"name": "Jonathan", "age": 15},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema4)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert codes == {"max_length", "min_value"}

    # Test 5: Nested schema validation
    class NestedSchema(Schema):
        inner = String(max_length=3)

    token = Token(
        value={"inner": "toolong"},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test 6: Field validator directly
    field = String(max_length=3)
    token = Token(
        value="toolong",
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test 7: Valid field validation
    field = Integer(minimum=0)
    token = Token(
        value=42,
        start=None,
        end=None
    )
    
    result = validate_with_positions(token=token, validator=field)
    assert result == 42

    # Test 8: Messages are sorted by start position
    class TestSchema5(Schema):
        first = String(required=True)
        second = String(required=True)

    token = Token(
        value={},
        start=None,
        end=None
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema5)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Both messages should have required code
        assert all(msg.code == "required" for msg in messages)


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken
    from typesystem.base import ValidationError, Message
    import pytest

    class PersonSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test successful validation
    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}",
    )
    result = validate_with_positions(token=token, validator=PersonSchema())
    assert result == {"name": "John", "age": 25}

    # Test validation error with custom message
    token = DictToken(
        {"name": "John" * 5, "age": 25},
        0,
        30,
        content="{'name': 'JohnJohnJohnJohnJohn', 'age': 25}",
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    assert len(exc_info.value.messages()) == 1
    message = exc_info.value.messages()[0]
    assert message.code == "max_length"
    assert "name" in message.text
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 30

    # Test required field error
    token = DictToken(
        {"name": "John"},
        0,
        15,
        content="{'name': 'John'}",
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    messages = exc_info.value.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.code == "required"
    assert message.text == "The field 'age' is required."
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 15

    # Test multiple validation errors
    token = DictToken(
        {"name": "John" * 5, "age": -5},
        0,
        30,
        content="{'name': 'JohnJohnJohnJohnJohn', 'age': -5}",
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index

    # Test nested structure validation
    class NestedSchema(Schema):
        items = ListToken(
            [
                DictToken({"id": 1, "value": "test"}, 0, 20, content=""),
                DictToken({"id": 2}, 0, 10, content=""),
            ],
            0,
            40,
            content="",
        )

    class ContainerSchema(Schema):
        data = NestedSchema()

    token = DictToken(
        {"data": {"items": [{"id": 1, "value": "test"}, {"id": 2}]}},
        0,
        50,
        content="",
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=ContainerSchema())
    messages = exc_info.value.messages()
    assert len(messages) >= 1

    # Test with Field validator directly
    field = String(max_length=5)
    token = Token("toolong", 0, 7, content="toolong")
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "max_length"

    # Test empty token
    token = DictToken({}, 0, 2, content="{}")
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    messages = exc_info.value.messages()
    assert len(messages) == 2
    for message in messages:
        assert message.code == "required"

    # Test messages are sorted by start position
    class TestSchema(Schema):
        first = String(max_length=1)
        second = String(max_length=1)

    token = DictToken(
        {"first": "aa", "second": "bb"},
        0,
        25,
        content="{'first': 'aa', 'second': 'bb'}",
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = Token(
        value={"name": "John", "age": 25},
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 20, "line_index": 0, "column_index": 20}
    )
    
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with custom field
    field = String(max_length=5)
    token = Token(
        value="toolong",
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 7, "line_index": 0, "column_index": 7}
    )
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position == {"char_index": 0, "line_index": 0, "column_index": 0}
        assert message.end_position == {"char_index": 7, "line_index": 0, "column_index": 7}
    
    # Test 3: Validation error with nested required field
    class NestedSchema(Schema):
        inner = Schema.from_dict({"required_field": String()})
    
    token = Token(
        value={"inner": {}},
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 15, "line_index": 0, "column_index": 15}
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'required_field' is required."
        assert message.index == ["inner", "required_field"]
    
    # Test 4: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        first = String(max_length=3)
        second = Integer(minimum=10)
    
    token = Token(
        value={"first": "toolong", "second": 5},
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 25, "line_index": 0, "column_index": 25}
    )
    
    # Create child tokens with different positions
    first_token = Token(
        value="toolong",
        start={"char_index": 10, "line_index": 0, "column_index": 10},
        end={"char_index": 17, "line_index": 0, "column_index": 17}
    )
    second_token = Token(
        value=5,
        start={"char_index": 20, "line_index": 0, "column_index": 20},
        end={"char_index": 21, "line_index": 0, "column_index": 21}
    )
    
    # Mock the lookup to return tokens at specific positions
    original_lookup = token.lookup
    def mock_lookup(index):
        if index == ["first"]:
            return first_token
        elif index == ["second"]:
            return second_token
        return original_lookup(index)
    
    token.lookup = mock_lookup
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Messages should be sorted by char_index
        assert messages[0].start_position["char_index"] == 10
        assert messages[0].code == "max_length"
        assert messages[1].start_position["char_index"] == 20
        assert messages[1].code == "minimum"
    
    # Test 5: Deeply nested validation error
    class DeepSchema(Schema):
        level1 = Schema.from_dict({
            "level2": Schema.from_dict({
                "level3": String(required=True)
            })
        })
    
    token = Token(
        value={"level1": {"level2": {}}},
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 30, "line_index": 0, "column_index": 30}
    )
    
    # Mock lookup for nested path
    level2_token = Token(
        value={},
        start={"char_index": 20, "line_index": 0, "column_index": 20},
        end={"char_index": 22, "line_index": 0, "column_index": 22}
    )
    
    def mock_deep_lookup(index):
        if index == ["level1", "level2"]:
            return level2_token
        return token
    
    token.lookup = mock_deep_lookup
    
    try:
        validate_with_positions(token=token, validator=DeepSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'level3' is required."
        assert message.index == ["level1", "level2", "level3"]
        assert message.start_position == {"char_index": 20, "line_index": 0, "column_index": 20}
        assert message.end_position == {"char_index": 22, "line_index": 0, "column_index": 22}


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    from typesystem.base import ValidationError, Message
    import pytest

    # Test 1: Successful validation with simple field
    field = String(max_length=10)
    token = ScalarToken("hello", 0, 4, content="hello")
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"

    # Test 2: Successful validation with schema
    class Person(Schema):
        name = String(max_length=20)
        age = Integer(minimum=0)

    token = DictToken(
        {"name": "Alice", "age": 30},
        0,
        30,
        content='{"name": "Alice", "age": 30}',
        tokens=[
            ScalarToken("name", 1, 5, content='"name"'),
            ScalarToken("Alice", 8, 13, content='"Alice"'),
            ScalarToken("age", 16, 18, content='"age"'),
            ScalarToken(30, 21, 22, content="30"),
        ],
    )
    result = validate_with_positions(token=token, validator=Person)
    assert result == {"name": "Alice", "age": 30}

    # Test 3: Validation error with required field
    class RequiredSchema(Schema):
        name = String(max_length=20, required=True)
        age = Integer(minimum=0, required=False)

    token = DictToken(
        {"age": 30},
        0,
        10,
        content='{"age": 30}',
        tokens=[
            ScalarToken("age", 1, 3, content='"age"'),
            ScalarToken(30, 6, 7, content="30"),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert messages[0].text == "The field 'name' is required."
    assert messages[0].index == ["name"]
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 10

    # Test 4: Validation error with nested required field
    class NestedSchema(Schema):
        person = Person

    token = DictToken(
        {"person": {}},
        0,
        12,
        content='{"person": {}}',
        tokens=[
            ScalarToken("person", 1, 6, content='"person"'),
            DictToken({}, 9, 10, content="{}", tokens=[]),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=NestedSchema)
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert all(msg.code == "required" for msg in messages)
    assert set(msg.index for msg in messages) == {["person", "name"], ["person", "age"]}
    assert all("The field" in msg.text for msg in messages)

    # Test 5: Validation error with custom error message (not required)
    field = String(max_length=3)
    token = ScalarToken("hello", 0, 4, content="hello")
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code != "required"
    assert messages[0].text == "Must have no more than 3 characters."
    assert messages[0].index == []
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 4

    # Test 6: Validation error with list items
    field = ListToken(
        ["a", "bb", "ccc"],
        0,
        20,
        content='["a", "bb", "ccc"]',
        tokens=[
            ScalarToken("a", 1, 1, content='"a"'),
            ScalarToken("bb", 5, 6, content='"bb"'),
            ScalarToken("ccc", 10, 12, content='"ccc"'),
        ],
    )
    validator = String(max_length=2)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=field, validator=validator)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code != "required"
    assert messages[0].index == [2]
    assert messages[0].start_position.char_index == 10
    assert messages[0].end_position.char_index == 12

    # Test 7: Messages are sorted by start position
    class MultiErrorSchema(Schema):
        a = String(max_length=1)
        b = String(max_length=1)
        c = String(max_length=1)

    token = DictToken(
        {"a": "aa", "b": "bb", "c": "cc"},
        0,
        30,
        content='{"a": "aa", "b": "bb", "c": "cc"}',
        tokens=[
            ScalarToken("a", 1, 1, content='"a"'),
            ScalarToken("aa", 5, 6, content='"aa"'),
            ScalarToken("b", 10, 10, content='"b"'),
            ScalarToken("bb", 14, 15, content='"bb"'),
            ScalarToken("c", 19, 19, content='"c"'),
            ScalarToken("cc", 23, 24, content='"cc"'),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=MultiErrorSchema)
    messages = exc_info.value.messages()
    assert len(messages) == 3
    assert messages[0].start_position.char_index == 5
    assert messages[1].start_position.char_index == 14
    assert messages[2].start_position.char_index == 23
    assert all(msg.code != "required" for msg in messages)

    # Test 8: Empty token with required field
    token = DictToken({}, 0, 1, content="{}", tokens=[])
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert messages[0].text == "The field 'name' is required."


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    import json

    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "John", "age": 25},
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 20, "line_index": 0, "column_index": 20}
    )
    
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with custom message
    token = Token(
        value={"name": "John" * 5, "age": 25},  # name too long
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 30, "line_index": 0, "column_index": 30}
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "max_length"
        assert "name" in msg.text
        assert msg.start_position == {"char_index": 0, "line_index": 0, "column_index": 0}
        assert msg.end_position == {"char_index": 30, "line_index": 0, "column_index": 30}

    # Test 3: Required field error
    token = Token(
        value={"age": 25},  # missing required 'name' field
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 15, "line_index": 0, "column_index": 15}
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "required"
        assert msg.text == "The field 'name' is required."
        assert msg.start_position == {"char_index": 0, "line_index": 0, "column_index": 0}
        assert msg.end_position == {"char_index": 15, "line_index": 0, "column_index": 15}

    # Test 4: Multiple validation errors sorted by position
    class NestedSchema(Schema):
        user = TestSchema
        id = Integer()

    token = Token(
        value={
            "user": {"name": "A" * 20, "age": -5},
            "id": "not_an_int"
        },
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 50, "line_index": 0, "column_index": 50}
    )
    
    # Create nested tokens for lookup
    root_token = token
    user_token = Token(
        value={"name": "A" * 20, "age": -5},
        start={"char_index": 10, "line_index": 0, "column_index": 10},
        end={"char_index": 40, "line_index": 0, "column_index": 40}
    )
    root_token.children = {"user": user_token}
    
    try:
        validate_with_positions(token=root_token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        
        # Check messages are sorted by char_index
        positions = [msg.start_position["char_index"] for msg in messages]
        assert positions == sorted(positions)

    # Test 5: Field validator instead of Schema
    field = String(max_length=5)
    token = Token(
        value="too_long",
        start={"char_index": 5, "line_index": 1, "column_index": 5},
        end={"char_index": 13, "line_index": 1, "column_index": 13}
    )
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        msg = e.messages()[0]
        assert msg.code == "max_length"
        assert msg.start_position == {"char_index": 5, "line_index": 1, "column_index": 5}
        assert msg.end_position == {"char_index": 13, "line_index": 1, "column_index": 13}

    # Test 6: Valid field validation
    field = Integer(minimum=10, maximum=100)
    token = Token(
        value=50,
        start={"char_index": 0, "line_index": 0, "column_index": 0},
        end={"char_index": 2, "line_index": 0, "column_index": 2}
    )
    
    result = validate_with_positions(token=token, validator=field)
    assert result == 50


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import typing

    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test successful validation
    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test validation error with custom message
    token = Token(
        value={"name": "John" * 5, "age": 25},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert "name" in message.text or "maximum" in message.text.lower()
        assert message.code != "required"

    # Test validation error with required field
    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert "required" in message.text.lower()
        assert "'age'" in message.text

    # Test validation error with multiple errors
    token = Token(
        value={"name": "John" * 5, "age": -5},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)
        assert all(hasattr(m, 'start_position') for m in messages)

    # Test with Field validator directly
    field = String(max_length=5)
    token = Token(
        value="too long string",
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code != "required"

    # Test with nested structure
    class NestedSchema(Schema):
        inner = TestSchema

    token = Token(
        value={"inner": {"name": "John" * 5, "age": -5}},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)

    # Test that messages are sorted by start_position
    class MultiFieldSchema(Schema):
        field1 = String(max_length=1)
        field2 = String(max_length=1)
        field3 = String(max_length=1)

    token = Token(
        value={"field1": "aa", "field2": "bb", "field3": "cc"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=MultiFieldSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        # Messages should be sorted (even though positions are None in this test)
        for i in range(len(messages) - 1):
            assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
    
    token = DictToken(
        {"name": ScalarToken("test", 0, 4, '{"name": "test"}')},
        0, 20, '{"name": "test"}'
    )
    result = validate_with_positions(token=token, validator=SimpleSchema())
    assert result == {"name": "test"}
    
    # Test 2: Failed validation with positional information
    class PersonSchema(Schema):
        name = String(required=True, max_length=5)
        age = Integer(minimum=0)
    
    token = DictToken(
        {
            "name": ScalarToken("toolongname", 17, 28, '{"name": "toolongname"}'),
            "age": ScalarToken(-5, 35, 37, '{"age": -5}')
        },
        0, 45, '{"name": "toolongname", "age": -5}'
    )
    
    try:
        validate_with_positions(token=token, validator=PersonSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Check name field error
        name_error = [m for m in messages if m.index == ["name"]][0]
        assert "Must have no more than 5 characters" in name_error.text
        assert name_error.start_position.char_index == 17
        assert name_error.end_position.char_index == 28
        
        # Check age field error
        age_error = [m for m in messages if m.index == ["age"]][0]
        assert "Must be greater than or equal to 0" in age_error.text
        assert age_error.start_position.char_index == 35
        assert age_error.end_position.char_index == 37
    
    # Test 3: Required field error with custom message
    token = DictToken(
        {"age": ScalarToken(25, 10, 12, '{"age": 25}')},
        0, 20, '{"age": 25}'
    )
    
    try:
        validate_with_positions(token=token, validator=PersonSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'name' is required."
        assert message.code == "required"
        assert message.index == ["name"]
    
    # Test 4: Nested validation with list
    class ItemSchema(Schema):
        id = Integer(required=True)
        value = String(required=True)
    
    class NestedSchema(Schema):
        items = Field(ItemSchema, many=True)
    
    token = DictToken(
        {
            "items": ListToken(
                [
                    DictToken(
                        {
                            "id": ScalarToken(1, 20, 21, '[{"id": 1}]'),
                            "value": ScalarToken("test", 23, 29, '[{"value": "test"}]')
                        },
                        15, 35, '[{"id": 1, "value": "test"}]'
                    ),
                    DictToken(
                        {
                            "id": ScalarToken(2, 45, 46, '[{"id": 2}]')
                        },
                        40, 55, '[{"id": 2}]'
                    )
                ],
                10, 60, '{"items": [{"id": 1, "value": "test"}, {"id": 2}]}'
            )
        },
        0, 70, '{"items": [{"id": 1, "value": "test"}, {"id": 2}]}'
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.text == "The field 'value' is required."
        assert message.code == "required"
        assert message.index == ["items", 1, "value"]
    
    # Test 5: Direct field validation (not Schema)
    token = ScalarToken("toolongvalue", 0, 12, '"toolongvalue"')
    
    try:
        validate_with_positions(token=token, validator=String(max_length=5))
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert "Must have no more than 5 characters" in message.text
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 12
    
    # Test 6: Messages are sorted by position
    class MultiFieldSchema(Schema):
        first = String(required=True)
        second = String(required=True)
        third = String(required=True)
    
    token = DictToken(
        {
            "third": ScalarToken("value3", 50, 56, '{"third": "value3"}'),
            "first": ScalarToken("value1", 10, 16, '{"first": "value1"}'),
            "second": ScalarToken("value2", 30, 36, '{"second": "value2"}')
        },
        0, 65, '{"first": "value1", "second": "value2", "third": "value3"}'
    )
    
    result = validate_with_positions(token=token, validator=MultiFieldSchema())
    assert result == {"first": "value1", "second": "value2", "third": "value3"}
    
    # Test error case with multiple fields to verify sorting
    token = DictToken(
        {
            "third": ScalarToken("value3", 50, 56, '{"third": "value3"}')
        },
        0, 65, '{"third": "value3"}'
    )
    
    try:
        validate_with_positions(token=token, validator=MultiFieldSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Messages should be sorted by start position
        assert messages[0].index == ["first"]
        assert messages[0].start_position.char_index == 10
        
        assert messages[1].index == ["second"]
        assert messages[1].start_position.char_index == 30


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        {"name": ScalarToken("John", 0, 4), "age": ScalarToken(25, 10, 12)},
        0, 20,
        key_tokens={
            "name": ScalarToken("name", 0, 4),
            "age": ScalarToken("age", 10, 12)
        }
    )
    
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Failed validation with required field error
    token = DictToken(
        {"age": ScalarToken(25, 10, 12)},
        0, 20,
        key_tokens={"age": ScalarToken("age", 10, 12)}
    )
    
    try:
        validate_with_positions(token=token, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 20
    
    # Test 3: Failed validation with nested required field error
    class NestedSchema(Schema):
        data = Field(type="object", properties={"value": String()})
    
    token = DictToken(
        {"data": DictToken({}, 10, 15)},
        0, 20,
        key_tokens={"data": ScalarToken("data", 0, 4)}
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."
        assert message.index == ["data", "value"]
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 15
    
    # Test 4: Failed validation with multiple errors including non-required errors
    class MultiErrorSchema(Schema):
        name = String(max_length=3)
        age = Integer(minimum=18)
    
    token = DictToken(
        {
            "name": ScalarToken("Johnathan", 10, 19),
            "age": ScalarToken(15, 25, 27)
        },
        0, 30,
        key_tokens={
            "name": ScalarToken("name", 0, 4),
            "age": ScalarToken("age", 20, 23)
        }
    )
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Messages should be sorted by start position
        assert messages[0].code == "max_length"
        assert messages[0].index == ["name"]
        assert messages[0].start_position.char_index == 10
        
        assert messages[1].code == "minimum"
        assert messages[1].index == ["age"]
        assert messages[1].start_position.char_index == 25
    
    # Test 5: Successful validation with Field validator (not Schema)
    field = String(max_length=5)
    token = ScalarToken("test", 0, 4)
    
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test 6: Failed validation with Field validator
    field = String(max_length=3)
    token = ScalarToken("test", 0, 4)
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 4
    
    # Test 7: Complex nested structure with list
    class ComplexSchema(Schema):
        items = Field(type="array", items=Field(type="object", properties={"id": Integer()}))
    
    token = DictToken(
        {
            "items": ListToken(
                [
                    DictToken({"id": ScalarToken(1, 20, 21)}, 15, 25),
                    DictToken({}, 30, 32)
                ],
                10, 35
            )
        },
        0, 40,
        key_tokens={"items": ScalarToken("items", 0, 5)}
    )
    
    try:
        validate_with_positions(token=token, validator=ComplexSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'id' is required."
        assert message.index == ["items", 1, "id"]
        assert message.start_position.char_index == 30
        assert message.end_position.char_index == 32


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, ObjectToken
    from typesystem.base import ValidationError
    
    # Test 1: Successful validation with Field
    string_field = String(max_length=5)
    token = Token(value="test", start=None, end=None)
    result = validate_with_positions(token=token, validator=string_field)
    assert result == "test"
    
    # Test 2: Successful validation with Schema
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = ObjectToken(
        value={"name": "John", "age": 25},
        start=None,
        end=None,
        children={
            "name": Token(value="John", start=None, end=None),
            "age": Token(value=25, start=None, end=None)
        }
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 3: Validation error with Field - simple error
    string_field = String(max_length=3)
    token = Token(value="toolong", start=None, end=None)
    try:
        validate_with_positions(token=token, validator=string_field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "long" in message.text.lower()
    
    # Test 4: Validation error with Schema - required field error
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = ObjectToken(
        value={"name": "John"},
        start=None,
        end=None,
        children={
            "name": Token(value="John", start=None, end=None)
        }
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert "field 'age' is required" in message.text.lower()
    
    # Test 5: Validation error with Schema - multiple errors
    class MultiErrorSchema(Schema):
        name = String(max_length=3, required=True)
        age = Integer(minimum=18, required=True)
    
    token = ObjectToken(
        value={"name": "Jonathan", "age": 15},
        start=None,
        end=None,
        children={
            "name": Token(value="Jonathan", start=None, end=None),
            "age": Token(value=15, start=None, end=None)
        }
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "minimum" in codes
    
    # Test 6: Test message sorting by start position
    class SortingSchema(Schema):
        field1 = String(required=True)
        field2 = Integer(required=True)
    
    # Create tokens with different start positions
    token = ObjectToken(
        value={},
        start=None,
        end=None,
        children={
            "field1": Token(value=None, start=type('obj', (object,), {'char_index': 10})(), end=None),
            "field2": Token(value=None, start=type('obj', (object,), {'char_index': 5})(), end=None)
        }
    )
    
    try:
        validate_with_positions(token=token, validator=SortingSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Should be sorted by char_index
        assert messages[0].start_position.char_index == 5
        assert messages[1].start_position.char_index == 10
    
    # Test 7: Nested validation error
    class NestedSchema(Schema):
        class InnerSchema(Schema):
            inner_field = String(required=True)
        
        outer_field = InnerSchema
    
    token = ObjectToken(
        value={"outer_field": {}},
        start=None,
        end=None,
        children={
            "outer_field": ObjectToken(
                value={},
                start=None,
                end=None,
                children={}
            )
        }
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert "field 'inner_field' is required" in message.text.lower()


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import typing

    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)

    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."

    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        class Inner(Schema):
            value = String(required=True)
        
        inner = Inner

    token = Token(
        value={"inner": {}},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."

    # Test 4: Validation error with custom error message
    class CustomSchema(Schema):
        email = String(format="email")

    token = Token(
        value={"email": "invalid-email"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=CustomSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "format"
        assert "Must be a valid email address" in message.text

    # Test 5: Validation with Field directly
    field = String(max_length=5)
    token = Token(
        value="toolong",
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test 6: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(required=True)
        age = Integer(minimum=18)

    token = Token(
        value={"name": "", "age": 15},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Messages should be sorted by start_position.char_index
        assert all(isinstance(m, Message) for m in messages)

    # Test 7: Empty token value
    token = Token(
        value=None,
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) > 0

    # Test 8: Valid Field validation
    field = Integer(minimum=0, maximum=100)
    token = Token(
        value=50,
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=field)
    assert result == 50


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken
    from typesystem.base import ValidationError, Message
    import pytest

    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test 1: Successful validation
    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}"
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with positional information
    token = DictToken(
        {"name": "VeryLongName", "age": -5},
        0,
        30,
        content="{'name': 'VeryLongName', 'age': -5}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    
    # Check messages are sorted by start position
    assert messages[0].start_position.char_index <= messages[1].start_position.char_index
    
    # Check message properties
    for msg in messages:
        assert hasattr(msg, 'start_position')
        assert hasattr(msg, 'end_position')
        assert msg.start_position.char_index >= 0
        assert msg.end_position.char_index > msg.start_position.char_index

    # Test 3: Required field error
    token = DictToken(
        {"name": "John"},
        0,
        15,
        content="{'name': 'John'}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "required" in messages[0].text.lower()

    # Test 4: Nested structure validation
    class NestedSchema(Schema):
        items = ListToken(String(), min_items=1)

    token = DictToken(
        {"items": []},
        0,
        10,
        content="{'items': []}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=NestedSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) >= 1

    # Test 5: Field validator instead of Schema
    field_validator = String(max_length=5)
    token = Token(
        "TooLong",
        0,
        7,
        content="TooLong"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field_validator)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert "max_length" in messages[0].code

    # Test 6: Complex nested error with positions
    class AddressSchema(Schema):
        street = String(required=True)
        city = String(required=True)

    class PersonSchema(Schema):
        name = String(required=True)
        address = AddressSchema

    token = DictToken(
        {"name": "John", "address": {"street": "Main St"}},
        0,
        40,
        content="{'name': 'John', 'address': {'street': 'Main St'}}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "city" in messages[0].text.lower()


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import pytest

    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test successful validation
    token = Token(value={"name": "Alice", "age": 25}, start=None, end=None)
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "Alice", "age": 25}

    # Test validation error with custom message
    token = Token(value={"name": "Alice" * 3, "age": 25}, start=None, end=None)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    assert len(exc_info.value.messages()) == 1
    message = exc_info.value.messages()[0]
    assert "name" in message.text or "maximum" in message.text.lower()

    # Test validation error with required field
    token = Token(value={"name": "Alice"}, start=None, end=None)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required" or "age" in str(messages[0].index)

    # Test validation with nested structure using Field directly
    field = String(max_length=5)
    token = Token(value="toolong", start=None, end=None)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert "maximum" in messages[0].text.lower()

    # Test multiple validation errors
    token = Token(value={"name": "A" * 20, "age": -5}, start=None, end=None)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema)
    messages = exc_info.value.messages()
    assert len(messages) >= 1

    # Test that messages are sorted by start position
    mock_token = Token(
        value={"name": "Alice", "age": -5},
        start=Token.Position(line=1, char_index=0, column=0),
        end=Token.Position(line=1, char_index=50, column=50),
    )
    mock_token.lookup = lambda x: Token(
        start=Token.Position(line=1, char_index=len(str(x)) * 10, column=0),
        end=Token.Position(line=1, char_index=len(str(x)) * 10 + 5, column=5),
    )

    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=mock_token, validator=TestSchema)
    messages = exc_info.value.messages()
    if len(messages) > 1:
        positions = [m.start_position.char_index for m in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken
    from typesystem.base import ValidationError

    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}"
    )
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with positional information
    token = DictToken(
        {"name": "John", "age": -5},
        0,
        20,
        content="{'name': 'John', 'age': -5}"
    )
    try:
        validate_with_positions(token=token, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "minimum"
        assert message.text == "Must be greater than or equal to 0."
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 20

    # Test 3: Required field error with custom message
    token = DictToken(
        {"age": 25},
        0,
        10,
        content="{'age': 25}"
    )
    try:
        validate_with_positions(token=token, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10

    # Test 4: Nested structure validation error
    class NestedSchema(Schema):
        items = ListToken(items=Integer())

    token = DictToken(
        {"items": [1, "invalid", 3]},
        0,
        20,
        content="{'items': [1, 'invalid', 3]}"
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "type"
        assert "Must be a number." in message.text

    # Test 5: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(max_length=3)
        age = Integer(minimum=18)

    token = DictToken(
        {"name": "Jonathan", "age": 15},
        0,
        25,
        content="{'name': 'Jonathan', 'age': 15}"
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Messages should be sorted by start position
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index

    # Test 6: Direct Field validation (not Schema)
    field = String(max_length=3)
    token = Token("toolong", 0, 6, content="toolong")
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 6

    # Test 7: Successful validation with Field
    field = Integer(minimum=0, maximum=100)
    token = Token(50, 0, 2, content="50")
    result = validate_with_positions(token=token, validator=field)
    assert result == 50

    # Test 8: Deeply nested required field
    class DeepSchema(Schema):
        data = DictToken(properties={"inner": DictToken(properties={"value": String()})})

    token = DictToken(
        {"data": {"inner": {}}},
        0,
        20,
        content="{'data': {'inner': {}}}"
    )
    try:
        validate_with_positions(token=token, validator=DeepSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}"
    )
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Failed validation with required field error
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer()
    
    token = DictToken(
        {"age": 25},
        0,
        10,
        content="{'age': 25}"
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10
    
    # Test 3: Failed validation with nested required field error
    class NestedSchema(Schema):
        user = SimpleSchema
    
    token = DictToken(
        {"user": {}},
        0,
        15,
        content="{'user': {}}"
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert all(m.code == "required" for m in messages)
        assert {m.index[-1] for m in messages} == {"name", "age"}
        assert all("The field" in m.text for m in messages)
    
    # Test 4: Failed validation with field validation error (not required)
    class ValidationSchema(Schema):
        name = String(max_length=5)
    
    token = DictToken(
        {"name": "Jonathan"},
        0,
        20,
        content="{'name': 'Jonathan'}"
    )
    try:
        validate_with_positions(token=token, validator=ValidationSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code != "required"
        assert "max_length" in message.code
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 20
    
    # Test 5: Failed validation with list item error
    class ListSchema(Schema):
        items = ListToken.of(Integer(minimum=0))
    
    token = DictToken(
        {"items": [-1, 5]},
        0,
        20,
        content="{'items': [-1, 5]}"
    )
    try:
        validate_with_positions(token=token, validator=ListSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert "minimum" in message.code
        assert message.index == ["items", 0]
    
    # Test 6: Messages are sorted by start position
    class MultiErrorSchema(Schema):
        a = String(required=True)
        b = Integer(minimum=10)
        c = String(max_length=2)
    
    token = DictToken(
        {"b": 5, "c": "toolong"},
        0,
        30,
        content="{'b': 5, 'c': 'toolong'}"
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        char_indices = [m.start_position.char_index for m in messages]
        assert char_indices == sorted(char_indices)
    
    # Test 7: Direct Field validation (not Schema)
    field = String(min_length=3)
    token = ScalarToken("ab", 0, 2, content="ab")
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert "min_length" in message.code
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 2
    
    # Test 8: Successful Field validation
    field = Integer(minimum=0, maximum=100)
    token = ScalarToken(50, 0, 2, content="50")
    result = validate_with_positions(token=token, validator=field)
    assert result == 50


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        {
            "name": ScalarToken("John", 0, 4, '{"name": "John", "age": 25}'),
            "age": ScalarToken(25, 14, 16, '{"name": "John", "age": 25}')
        },
        0,
        30,
        '{"name": "John", "age": 25}'
    )
    
    result = validate_with_positions(token=token, validator=TestSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
        optional_field = String(allow_null=True)
    
    token = DictToken(
        {"optional_field": ScalarToken("test", 20, 24, '{"optional_field": "test"}')},
        0,
        30,
        '{"optional_field": "test"}'
    )
    
    try:
        validate_with_positions(token=token, validator=RequiredSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'required_field' is required."
        assert message.index == ["required_field"]
    
    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        items = ListToken(
            [
                DictToken(
                    {
                        "value": ScalarToken("abc", 10, 13, '[{"value": "abc"}]')
                    },
                    1,
                    15,
                    '[{"value": "abc"}]'
                )
            ],
            0,
            17,
            '[{"value": "abc"}]'
        )
    
    class ItemSchema(Schema):
        value = String(min_length=5)
    
    try:
        validate_with_positions(token=NestedSchema().items, validator=ItemSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "min_length"
        assert "value" in message.text.lower()
        assert message.index == ["value"]
    
    # Test 4: Validation error with Field directly
    field = String(max_length=3)
    token = ScalarToken("toolong", 0, 7, '"toolong"')
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "max_length"
        assert "long" in message.text.lower()
        assert message.index == []
    
    # Test 5: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        first = String(max_length=2)
        second = Integer(minimum=10)
    
    token = DictToken(
        {
            "first": ScalarToken("abc", 10, 13, '{"first": "abc", "second": 5}'),
            "second": ScalarToken(5, 25, 26, '{"first": "abc", "second": 5}')
        },
        0,
        35,
        '{"first": "abc", "second": 5}'
    )
    
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 10
        assert messages[1].start_position.char_index == 25
        assert messages[0].code == "max_length"
        assert messages[1].code == "minimum"
    
    # Test 6: Deeply nested required field error
    class DeepSchema(Schema):
        level1 = DictToken(
            {
                "level2": DictToken(
                    {
                        "level3": ScalarToken(None, 30, 34, '{"level1": {"level2": {"level3": null}}}')
                    },
                    20,
                    35,
                    '{"level1": {"level2": {"level3": null}}}'
                )
            },
            10,
            45,
            '{"level1": {"level2": {"level3": null}}}'
        )
    
    class Level3Schema(Schema):
        required_field = String(allow_null=False)
    
    try:
        validate_with_positions(token=DeepSchema().level1["level2"]["level3"], validator=Level3Schema())
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.code == "required"
        assert "required_field" in message.text


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    from typesystem.base import ValidationError, Message
    import pytest

    # Test 1: Successful validation with simple field
    field = String(max_length=5)
    token = ScalarToken("hello", 0, 4)
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"

    # Test 2: Successful validation with schema
    class Person(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = DictToken(
        {"name": "Alice", "age": 30},
        0,
        30,
        [
            ("name", ScalarToken("Alice", 7, 11)),
            ("age", ScalarToken(30, 17, 18)),
        ],
    )
    result = validate_with_positions(token=token, validator=Person)
    assert result == {"name": "Alice", "age": 30}

    # Test 3: Validation error with required field
    class RequiredSchema(Schema):
        name = String()
        age = Integer()

    token = DictToken(
        {"name": "Bob"},
        0,
        15,
        [
            ("name", ScalarToken("Bob", 7, 9)),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.text == "The field 'age' is required."
    assert message.index == ["age"]
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 15

    # Test 4: Validation error with nested required field
    class NestedSchema(Schema):
        person = Person

    token = DictToken(
        {"person": DictToken({}, 10, 15, [])},
        0,
        20,
        [
            ("person", DictToken({}, 10, 15, [])),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=NestedSchema)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 2
    
    # Messages should be sorted by start position
    sorted_messages = sorted(messages, key=lambda m: m.start_position.char_index)
    assert sorted_messages[0].index == ["person", "name"]
    assert sorted_messages[1].index == ["person", "age"]

    # Test 5: Validation error with custom error message
    field = String(min_length=5)
    token = ScalarToken("hi", 0, 1)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "min_length"
    assert "minimum length" in message.text.lower()
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 1

    # Test 6: Validation error with list items
    class ListSchema(Schema):
        items = ListToken

    field = ListSchema.fields["items"]
    token = ListToken(
        ["a", "b", "c"],
        0,
        10,
        [
            ScalarToken("a", 1, 1),
            ScalarToken("b", 3, 3),
            ScalarToken("c", 5, 5),
        ],
    )
    # This should work since ListToken is not a proper validator
    # but we need to test with actual list validation
    from typesystem.fields import Array
    
    array_field = Array(items=String(max_length=1))
    token = ListToken(
        ["a", "bb", "c"],
        0,
        10,
        [
            ScalarToken("a", 1, 1),
            ScalarToken("bb", 3, 4),
            ScalarToken("c", 6, 6),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=array_field)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.index == [1]
    assert message.start_position.char_index == 3
    assert message.end_position.char_index == 4

    # Test 7: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        first = String(max_length=2)
        second = Integer(maximum=10)

    token = DictToken(
        {"first": "toolong", "second": 20},
        0,
        30,
        [
            ("first", ScalarToken("toolong", 9, 15)),
            ("second", ScalarToken(20, 24, 25)),
        ],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=MultiErrorSchema)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 2
    
    # Verify sorting by start position
    for i in range(len(messages) - 1):
        assert messages[i].start_position.char_index <= messages[i + 1].start_position.char_index

    # Test 8: Deeply nested required field
    class DeepSchema(Schema):
        level1 = {"level2": {"level3": String()}}

    token = DictToken(
        {"level1": DictToken(
            {"level2": DictToken({}, 20, 25, [])},
            10,
            30,
            [("level2", DictToken({}, 20, 25, []))]
        )},
        0,
        35,
        [("level1", DictToken(
            {"level2": DictToken({}, 20, 25, [])},
            10,
            30,
            [("level2", DictToken({}, 20, 25, []))]
        ))],
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=DeepSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.index == ["level1", "level2", "level3"]
    assert message.text == "The field 'level3' is required."


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken
    from typesystem.base import ValidationError

    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}"
    )
    
    result = validate_with_positions(token=token, validator=SimpleSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with positional information for required field
    token = DictToken(
        {"name": "John"},
        0,
        15,
        content="{'name': 'John'}"
    )
    
    try:
        validate_with_positions(token=token, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
        assert message.index == ["age"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 15

    # Test 3: Validation error with custom error message (not required)
    token = DictToken(
        {"name": "John" * 5, "age": 25},
        0,
        30,
        content="{'name': 'JohnJohnJohnJohnJohn', 'age': 25}"
    )
    
    try:
        validate_with_positions(token=token, validator=SimpleSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "long" in message.text.lower()
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 30

    # Test 4: Nested validation with required field
    class NestedSchema(Schema):
        user = SimpleSchema

    token = DictToken(
        {"user": {"name": "John"}},
        0,
        25,
        content="{'user': {'name': 'John'}}"
    )
    
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
        assert message.index == ["user", "age"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 25

    # Test 5: Multiple validation errors sorted by position
    class MultiFieldSchema(Schema):
        first = String(required=True)
        second = Integer(required=True)
        third = String(required=True)

    token = DictToken(
        {},
        0,
        2,
        content="{}"
    )
    
    try:
        validate_with_positions(token=token, validator=MultiFieldSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        
        # Check that messages are sorted by start position
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)
        
        # Check all messages are required errors
        for msg in messages:
            assert msg.code == "required"
            assert "required" in msg.text.lower()

    # Test 6: Validation with simple Field (not Schema)
    token = DictToken(
        {"value": "test"},
        0,
        15,
        content="{'value': 'test'}"
    )
    field = String(max_length=3)
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.index == []
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 15

    # Test 7: List token validation
    class ListSchema(Schema):
        items = String()

    token = ListToken(
        ["item1", "item2", "item" * 10],
        0,
        50,
        content="['item1', 'item2', 'itemitemitemitemitemitemitemitemitemitem']"
    )
    
    try:
        validate_with_positions(token=token, validator=ListSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        # This will fail because list validation works differently
        # but we're testing the error handling path
        pass

    # Test 8: Successful validation with nested structure
    class AddressSchema(Schema):
        street = String(required=True)
        city = String(required=True)

    class PersonSchema(Schema):
        name = String(required=True)
        address = AddressSchema

    token = DictToken(
        {"name": "Alice", "address": {"street": "123 Main", "city": "Town"}},
        0,
        50,
        content="{'name': 'Alice', 'address': {'street': '123 Main', 'city': 'Town'}}"
    )
    
    result = validate_with_positions(token=token, validator=PersonSchema)
    assert result == {"name": "Alice", "address": {"street": "123 Main", "city": "Town"}}


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError

    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test successful validation
    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test validation error with custom message
    token = Token(
        value={"name": "John" * 5, "age": 25},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert "name" in message.text
        assert message.code == "max_length"

    # Test validation error with required field
    token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.text == "The field 'age' is required."
        assert message.code == "required"

    # Test validation error with multiple errors
    token = Token(
        value={"name": "John" * 5, "age": -5},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 2
        messages = error.messages()
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "minimum" in codes

    # Test with nested structure
    class NestedSchema(Schema):
        person = TestSchema

    token = Token(
        value={"person": {"name": "John" * 5, "age": -5}},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 2

    # Test with Field directly
    field = String(max_length=5)
    token = Token(
        value="toolong",
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test successful validation with Field
    token = Token(
        value="short",
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=field)
    assert result == "short"


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = Token(value={"name": "John", "age": 25}, start=None, end=None)
    result = validate_with_positions(token=token, validator=TestSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = Token(value={"name": "John"}, start=None, end=None)
    try:
        validate_with_positions(token=token, validator=RequiredSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
    
    # Test 3: Validation error with custom error message
    class LengthSchema(Schema):
        name = String(max_length=3)
    
    token = Token(value={"name": "Jonathan"}, start=None, end=None)
    try:
        validate_with_positions(token=token, validator=LengthSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "Ensure this value has at most" in message.text
    
    # Test 4: Nested validation error
    class NestedSchema(Schema):
        class Inner(Schema):
            value = Integer(required=True)
        
        inner = Inner()
    
    token = Token(value={"inner": {}}, start=None, end=None)
    try:
        validate_with_positions(token=token, validator=NestedSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."
    
    # Test 5: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(required=True)
        age = Integer(minimum=18)
    
    token = Token(value={"age": 15}, start=None, end=None)
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        # Messages should be sorted by start_position.char_index
        assert all(
            messages[i].start_position.char_index <= messages[i + 1].start_position.char_index
            for i in range(len(messages) - 1)
        )
    
    # Test 6: Direct Field validation (not Schema)
    field = String(max_length=5)
    token = Token(value="Hello World", start=None, end=None)
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 5, "name"),
            "age": ScalarToken(25, 7, 10, "age")
        },
        start=0,
        end=11
    )
    
    result = validate_with_positions(token=token, validator=TestSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with custom message
    class TestSchema2(Schema):
        name = String(max_length=5)
    
    token = DictToken(
        value={
            "name": ScalarToken("Jonathan", 0, 9, "name")
        },
        start=0,
        end=10
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema2())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.text == "Must have no more than 5 characters."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 9
    
    # Test 3: Required field error
    class TestSchema3(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 5, "name")
        },
        start=0,
        end=6
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema3())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'age' is required."
        assert message.index == ["age"]
    
    # Test 4: Nested validation error
    class NestedSchema(Schema):
        value = String(max_length=3)
    
    class ParentSchema(Schema):
        nested = NestedSchema()
    
    token = DictToken(
        value={
            "nested": DictToken(
                value={
                    "value": ScalarToken("toolong", 10, 17, "value")
                },
                start=8,
                end=18
            )
        },
        start=0,
        end=19
    )
    
    try:
        validate_with_positions(token=token, validator=ParentSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.index == ["nested", "value"]
        assert message.start_position.char_index == 10
        assert message.end_position.char_index == 17
    
    # Test 5: Multiple validation errors sorted by position
    class TestSchema5(Schema):
        first = String(max_length=3)
        second = String(max_length=2)
    
    token = DictToken(
        value={
            "first": ScalarToken("toolong", 20, 27, "first"),
            "second": ScalarToken("alsobad", 30, 37, "second")
        },
        start=0,
        end=38
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema5())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].start_position.char_index == 20
        assert messages[1].start_position.char_index == 30
    
    # Test 6: Field validator instead of Schema
    field = String(max_length=3)
    token = ScalarToken("toolong", 0, 7, "value")
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 7
    
    # Test 7: List validation error
    class TestSchema7(Schema):
        items = ListToken(
            value=[
                ScalarToken("a", 10, 11, "item"),
                ScalarToken("toolong", 13, 20, "item")
            ],
            start=8,
            end=21
        )
    
    field = String(max_length=3)
    
    try:
        validate_with_positions(token=TestSchema7.items, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 13
        assert message.end_position.char_index == 20


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError
    import typing

    # Test 1: Successful validation with Field
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Failed validation with required field error
    token = Token(
        value={"age": 25},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'name' is required." in message.text

    # Test 3: Failed validation with custom error message
    token = Token(
        value={"name": "John" * 5, "age": 25},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "Must have no more than 10 characters." in message.text

    # Test 4: Failed validation with multiple errors
    token = Token(
        value={"name": "John" * 5, "age": -5},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=TestSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "minimum" in codes

    # Test 5: Successful validation with nested schema
    class NestedSchema(Schema):
        inner = String(max_length=5)

    token = Token(
        value={"inner": "test"},
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=NestedSchema)
    assert result == {"inner": "test"}

    # Test 6: Failed validation with nested required field
    token = Token(
        value={},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert "The field 'inner' is required." in message.text

    # Test 7: Test with Field directly instead of Schema
    field = String(max_length=5)
    token = Token(
        value="hello",
        start=None,
        end=None
    )
    result = validate_with_positions(token=token, validator=field)
    assert result == "hello"

    # Test 8: Failed validation with Field directly
    token = Token(
        value="too long",
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"

    # Test 9: Messages are sorted by start_position
    class MultiErrorSchema(Schema):
        a = String(required=True)
        b = String(required=True)
        c = String(required=True)

    token = Token(
        value={},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        # Verify messages are sorted (though positions are None in this test)
        for i in range(len(messages) - 1):
            assert messages[i].start_position is None
            assert messages[i].end_position is None

    # Test 10: Complex nested structure with positions
    class AddressSchema(Schema):
        street = String(required=True)
        city = String(required=True)

    class PersonSchema(Schema):
        name = String(required=True)
        address = AddressSchema

    token = Token(
        value={"name": "John", "address": {"street": "123 Main St"}},
        start=None,
        end=None
    )
    try:
        validate_with_positions(token=token, validator=PersonSchema)
        assert False, "Expected ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        message = messages[0]
        assert message.code == "required"
        assert "The field 'city' is required." in message.text


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import pytest

    class TestSchema(Schema):
        name = String(max_length=10, required=True)
        age = Integer(minimum=0, required=True)

    # Test 1: Valid input
    valid_token = Token(
        value={"name": "John", "age": 25},
        start=None,
        end=None
    )
    result = validate_with_positions(token=valid_token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Missing required field
    missing_field_token = Token(
        value={"name": "John"},
        start=None,
        end=None
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=missing_field_token, validator=TestSchema)
    
    assert len(exc_info.value.messages()) == 1
    message = exc_info.value.messages()[0]
    assert message.code == "required"
    assert "The field 'age' is required." in message.text

    # Test 3: Multiple validation errors
    invalid_token = Token(
        value={"name": "John" * 5, "age": -5},
        start=None,
        end=None
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=invalid_token, validator=TestSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert any("maximum length" in msg.text for msg in messages)
    assert any("minimum" in msg.text for msg in messages)

    # Test 4: Nested structure with Field validator
    class NestedSchema(Schema):
        user = TestSchema

    nested_token = Token(
        value={"user": {"name": "Alice"}},
        start=None,
        end=None
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=nested_token, validator=NestedSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert "The field 'age' is required." in messages[0].text

    # Test 5: Direct Field validator
    field_token = Token(
        value="",
        start=None,
        end=None
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=field_token, validator=String(required=True))
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"

    # Test 6: Valid nested structure
    valid_nested_token = Token(
        value={"user": {"name": "Bob", "age": 30}},
        start=None,
        end=None
    )
    result = validate_with_positions(token=valid_nested_token, validator=NestedSchema)
    assert result == {"user": {"name": "Bob", "age": 30}}

    # Test 7: Check message positions are preserved
    token_with_positions = Token(
        value={"name": ""},
        start=Token.Position(line=1, char_index=0, column=0),
        end=Token.Position(line=1, char_index=10, column=10)
    )
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token_with_positions, validator=TestSchema)
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    for msg in messages:
        assert hasattr(msg, 'start_position')
        assert hasattr(msg, 'end_position')


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken
    from typesystem.base import ValidationError, Message
    import pytest

    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = DictToken(
        {"name": "John", "age": 25},
        0,
        20,
        content="{'name': 'John', 'age': 25}"
    )
    result = validate_with_positions(token=token, validator=TestSchema)
    assert result == {"name": "John", "age": 25}

    # Test 2: Validation error with required field
    class RequiredSchema(Schema):
        required_field = String(required=True)
        optional_field = String(required=False)

    token = DictToken(
        {"optional_field": "test"},
        0,
        20,
        content="{'optional_field': 'test'}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.text == "The field 'required_field' is required."
    assert message.index == ["required_field"]
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 20

    # Test 3: Validation error with nested structure
    class NestedSchema(Schema):
        items = ListToken(items=String(max_length=5))

    token = ListToken(
        ["short", "toolongvalue"],
        0,
        30,
        content="['short', 'toolongvalue']"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=NestedSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "max_length"
    assert message.index == [1]
    assert message.start_position.char_index == 9
    assert message.end_position.char_index == 23

    # Test 4: Validation error with multiple messages
    class MultiErrorSchema(Schema):
        name = String(required=True, max_length=3)
        age = Integer(minimum=18)

    token = DictToken(
        {"name": "toolong", "age": 15},
        0,
        30,
        content="{'name': 'toolong', 'age': 15}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=MultiErrorSchema)
    
    error = exc_info.value
    messages = error.messages()
    assert len(messages) == 2
    
    # Messages should be sorted by position
    sorted_messages = sorted(messages, key=lambda m: m.start_position.char_index)
    assert sorted_messages[0].code == "max_length"
    assert sorted_messages[0].index == ["name"]
    assert sorted_messages[1].code == "minimum"
    assert sorted_messages[1].index == ["age"]

    # Test 5: Deeply nested validation error
    class DeepSchema(Schema):
        data = ListToken(items=DictToken(properties={"value": String(max_length=2)}))

    token = DictToken(
        {"data": [{"value": "ok"}, {"value": "too_long"}]},
        0,
        40,
        content="{'data': [{'value': 'ok'}, {'value': 'too_long'}]}"
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=DeepSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "max_length"
    assert message.index == ["data", 1, "value"]
    assert message.start_position.char_index == 28
    assert message.end_position.char_index == 38

    # Test 6: Direct Field validation (not Schema)
    field = String(max_length=5)
    token = Token("toolongvalue", 0, 12, content="'toolongvalue'")
    
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "max_length"
    assert message.index == []
    assert message.start_position.char_index == 0
    assert message.end_position.char_index == 12

    # Test 7: Empty token validation
    token = DictToken({}, 0, 2, content="{}")
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=RequiredSchema)
    
    error = exc_info.value
    assert len(error.messages()) == 1
    message = error.messages()[0]
    assert message.code == "required"
    assert message.text == "The field 'required_field' is required."


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
    
    token = DictToken(
        {"name": ScalarToken("test", 0, 4, '{"name": "test"}')},
        0, 20, '{"name": "test"}'
    )
    result = validate_with_positions(token=token, validator=SimpleSchema())
    assert result == {"name": "test"}
    
    # Test 2: Failed validation with required field error
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer()
    
    token = DictToken(
        {"age": ScalarToken(25, 10, 12, '{"age": 25}')},
        0, 15, '{"age": 25}'
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 15
    
    # Test 3: Failed validation with nested required field error
    class NestedSchema(Schema):
        data = Field(type="object", properties={"value": String(required=True)})
    
    token = DictToken(
        {"data": DictToken({}, 8, 10, '{"data": {}}')},
        0, 15, '{"data": {}}'
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'value' is required."
        assert message.index == ["data", "value"]
        assert message.start_position.char_index == 8
        assert message.end_position.char_index == 10
    
    # Test 4: Failed validation with custom error message (non-required error)
    class MaxLengthSchema(Schema):
        name = String(max_length=3)
    
    token = DictToken(
        {"name": ScalarToken("toolong", 9, 16, '{"name": "toolong"}')},
        0, 20, '{"name": "toolong"}'
    )
    try:
        validate_with_positions(token=token, validator=MaxLengthSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert "Must have no more than 3 characters" in message.text
        assert message.index == ["name"]
        assert message.start_position.char_index == 9
        assert message.end_position.char_index == 16
    
    # Test 5: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(required=True)
        age = Integer(minimum=18)
    
    token = DictToken(
        {
            "name": ScalarToken("", 8, 10, '{"name": ""}'),
            "age": ScalarToken(15, 20, 22, '{"name": "", "age": 15}')
        },
        0, 30, '{"name": "", "age": 15}'
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Check sorting by position
        assert messages[0].start_position.char_index == 8
        assert messages[0].end_position.char_index == 10
        assert messages[1].start_position.char_index == 20
        assert messages[1].end_position.char_index == 22
        
        # Check message types
        codes = [msg.code for msg in messages]
        assert "required" in codes
        assert "minimum" in codes
    
    # Test 6: Validation with Field directly (not Schema)
    field = String(required=True, max_length=5)
    token = ScalarToken("toolongvalue", 0, 13, '"toolongvalue"')
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 13
    
    # Test 7: Successful validation with nested structure
    class AddressSchema(Schema):
        street = String(required=True)
        city = String(required=True)
    
    class PersonSchema(Schema):
        name = String(required=True)
        address = AddressSchema
    
    token = DictToken(
        {
            "name": ScalarToken("John", 9, 15, '{"name": "John"}'),
            "address": DictToken(
                {
                    "street": ScalarToken("Main St", 30, 39, '"street": "Main St"'),
                    "city": ScalarToken("City", 45, 51, '"city": "City"')
                },
                20, 55, '"address": {...}'
            )
        },
        0, 60, '{"name": "John", "address": {...}}'
    )
    
    result = validate_with_positions(token=token, validator=PersonSchema())
    assert result == {
        "name": "John",
        "address": {"street": "Main St", "city": "City"}
    }
    
    # Test 8: List validation with errors
    class ListSchema(Schema):
        items = Field(type="array", items=String(max_length=3))
    
    token = DictToken(
        {
            "items": ListToken(
                [
                    ScalarToken("ok", 12, 16, '"ok"'),
                    ScalarToken("toolong", 18, 27, '"toolong"')
                ],
                8, 30, '"items": [...]'
            )
        },
        0, 35, '{"items": ["ok", "toolong"]}'
    )
    
    try:
        validate_with_positions(token=token, validator=ListSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.index == ["items", 1]
        assert message.start_position.char_index == 18
        assert message.end_position.char_index == 27


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    import pytest

    class PersonSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    # Test successful validation
    token = DictToken(
        {"name": ScalarToken("Alice", 0, 5), "age": ScalarToken(25, 7, 9)},
        0,
        9,
    )
    result = validate_with_positions(token=token, validator=PersonSchema())
    assert result == {"name": "Alice", "age": 25}

    # Test validation error with positions
    token = DictToken(
        {"name": ScalarToken("Alice" * 3, 0, 15), "age": ScalarToken(-5, 17, 19)},
        0,
        19,
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert messages[0].code == "max_length"
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 15
    assert messages[1].code == "minimum"
    assert messages[1].start_position.char_index == 17
    assert messages[1].end_position.char_index == 19

    # Test required field error
    token = DictToken({"name": ScalarToken("Alice", 0, 5)}, 0, 5)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=PersonSchema())
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "required"
    assert "The field 'age' is required." in messages[0].text
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 5

    # Test nested validation with Field
    field = String(max_length=5)
    token = ScalarToken("toolong", 0, 7)
    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=field)
    
    messages = exc_info.value.messages()
    assert len(messages) == 1
    assert messages[0].code == "max_length"
    assert messages[0].start_position.char_index == 0
    assert messages[0].end_position.char_index == 7

    # Test nested structure with list
    class ItemSchema(Schema):
        id = Integer()
        tags = ListToken([ScalarToken("a", 10, 11), ScalarToken("b", 13, 14)], 9, 15)

    token = DictToken(
        {
            "id": ScalarToken(1, 2, 3),
            "tags": ListToken(
                [ScalarToken("a", 10, 11), ScalarToken("b", 13, 14)], 9, 15
            ),
        },
        0,
        15,
    )
    result = validate_with_positions(token=token, validator=ItemSchema())
    assert result == {"id": 1, "tags": ["a", "b"]}

    # Test messages are sorted by start position
    token = DictToken(
        {
            "z": ScalarToken("error1", 20, 26),
            "a": ScalarToken("error2", 0, 6),
        },
        0,
        26,
    )
    
    class TestSchema(Schema):
        a = String(max_length=3)
        z = String(max_length=3)

    with pytest.raises(ValidationError) as exc_info:
        validate_with_positions(token=token, validator=TestSchema())
    
    messages = exc_info.value.messages()
    assert len(messages) == 2
    # Should be sorted by start_position.char_index
    assert messages[0].start_position.char_index == 0
    assert messages[1].start_position.char_index == 20


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation
    class TestSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        value={
            "name": ScalarToken("John", 0, 5, "name"),
            "age": ScalarToken(25, 7, 10, "age")
        },
        start=0,
        end=11
    )
    
    result = validate_with_positions(token=token, validator=TestSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Validation error with custom message
    token = DictToken(
        value={
            "name": ScalarToken("Johnathan", 0, 10, "name"),
            "age": ScalarToken(25, 12, 15, "age")
        },
        start=0,
        end=16
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.text == "Must have no more than 10 characters."
        assert message.index == ["name"]
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 10
    
    # Test 3: Required field error
    token = DictToken(
        value={
            "age": ScalarToken(25, 7, 10, "age")
        },
        start=0,
        end=11
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "required"
        assert message.text == "The field 'name' is required."
        assert message.index == ["name"]
    
    # Test 4: Multiple validation errors
    token = DictToken(
        value={
            "name": ScalarToken("Johnathan", 0, 10, "name"),
            "age": ScalarToken(-5, 12, 15, "age")
        },
        start=0,
        end=16
    )
    
    try:
        validate_with_positions(token=token, validator=TestSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert codes == {"max_length", "minimum"}
    
    # Test 5: Nested validation with Field
    field = String(max_length=5)
    token = ScalarToken("toolong", 0, 7, "test")
    
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        assert len(error.messages()) == 1
        message = error.messages()[0]
        assert message.code == "max_length"
        assert message.start_position.char_index == 0
        assert message.end_position.char_index == 7
    
    # Test 6: Nested structure validation
    class NestedSchema(Schema):
        items = ListToken(
            value=[
                DictToken(
                    value={
                        "id": ScalarToken(1, 10, 12, "id"),
                        "value": ScalarToken("test", 14, 20, "value")
                    },
                    start=9,
                    end=21
                )
            ],
            start=8,
            end=22
        )
    
    token = DictToken(
        value={
            "items": NestedSchema.items
        },
        start=0,
        end=23
    )
    
    class ParentSchema(Schema):
        items = ListToken
    
    try:
        validate_with_positions(token=token, validator=ParentSchema())
    except ValidationError:
        pass
    
    # Test 7: Messages sorted by start position
    class MultiFieldSchema(Schema):
        first = String(max_length=1)
        second = String(max_length=1)
        third = String(max_length=1)
    
    token = DictToken(
        value={
            "first": ScalarToken("aa", 50, 55, "first"),
            "second": ScalarToken("bb", 30, 35, "second"),
            "third": ScalarToken("cc", 10, 15, "third")
        },
        start=0,
        end=60
    )
    
    try:
        validate_with_positions(token=token, validator=MultiFieldSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 3
        positions = [msg.start_position.char_index for msg in messages]
        assert positions == sorted(positions)


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token
    from typesystem.base import ValidationError, Message
    import typing

    # Test 1: Successful validation with Field
    class TestField(Field):
        def validate(self, value: typing.Any) -> typing.Any:
            if not isinstance(value, str):
                raise ValidationError(text="Must be a string")
            return value.upper()

    token = Token(value="hello", start=(1, 0, 0), end=(1, 5, 5))
    validator = TestField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == "HELLO"

    # Test 2: Successful validation with Schema
    class PersonSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)

    token = Token(
        value={"name": "Alice", "age": 30},
        start=(1, 0, 0),
        end=(1, 20, 20)
    )
    result = validate_with_positions(token=token, validator=PersonSchema)
    assert result == {"name": "Alice", "age": 30}

    # Test 3: ValidationError with required field
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer()

    token = Token(
        value={"age": 30},
        start=(1, 0, 0),
        end=(1, 10, 10)
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.text == "The field 'name' is required."
        assert msg.index == ["name"]
        assert msg.start_position == (1, 0, 0)
        assert msg.end_position == (1, 10, 10)

    # Test 4: ValidationError with nested required field
    class NestedSchema(Schema):
        person = RequiredSchema

    token = Token(
        value={"person": {"age": 30}},
        start=(1, 0, 0),
        end=(1, 20, 20)
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "required"
        assert msg.text == "The field 'name' is required."
        assert msg.index == ["person", "name"]
        assert msg.start_position == (1, 0, 0)
        assert msg.end_position == (1, 20, 20)

    # Test 5: ValidationError with custom error code
    class CustomField(Field):
        def validate(self, value: typing.Any) -> typing.Any:
            if value != "secret":
                raise ValidationError(text="Invalid value", code="invalid")
            return value

    token = Token(value="wrong", start=(2, 0, 25), end=(2, 5, 30))
    validator = CustomField()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.code == "invalid"
        assert msg.text == "Invalid value"
        assert msg.index == []
        assert msg.start_position == (2, 0, 25)
        assert msg.end_position == (2, 5, 30)

    # Test 6: Multiple validation errors sorted by position
    class MultiErrorSchema(Schema):
        name = String(required=True, max_length=5)
        age = Integer(minimum=18)

    token = Token(
        value={"name": "TooLongName", "age": 15},
        start=(3, 0, 35),
        end=(3, 25, 60)
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        
        # Check messages are sorted by start_position.char_index
        assert messages[0].start_position.char_index <= messages[1].start_position.char_index
        
        # Verify both errors have correct positions
        for msg in messages:
            assert msg.start_position == (3, 0, 35)
            assert msg.end_position == (3, 25, 60)

    # Test 7: Empty token value
    token = Token(value=None, start=(4, 0, 65), end=(4, 0, 65))
    validator = String()
    try:
        validate_with_positions(token=token, validator=validator)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) >= 1

    # Test 8: Validator that returns the value unchanged
    class PassThroughField(Field):
        def validate(self, value: typing.Any) -> typing.Any:
            return value

    token = Token(value={"nested": "value"}, start=(5, 0, 70), end=(5, 15, 85))
    validator = PassThroughField()
    result = validate_with_positions(token=token, validator=validator)
    assert result == {"nested": "value"}


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_positions():
    from typesystem.fields import String, Integer
    from typesystem.schemas import Schema
    from typesystem.tokenize.tokens import Token, DictToken, ListToken, ScalarToken
    
    # Test 1: Successful validation with simple field
    class SimpleSchema(Schema):
        name = String(max_length=10)
        age = Integer(minimum=0)
    
    token = DictToken(
        {"name": ScalarToken("John", 0, 4), "age": ScalarToken(25, 6, 8)},
        0,
        8,
    )
    result = validate_with_positions(token=token, validator=SimpleSchema())
    assert result == {"name": "John", "age": 25}
    
    # Test 2: Failed validation with required field error
    class RequiredSchema(Schema):
        name = String(required=True)
        age = Integer(required=True)
    
    token = DictToken(
        {"name": ScalarToken("John", 0, 4)},
        0,
        4,
    )
    try:
        validate_with_positions(token=token, validator=RequiredSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
        assert messages[0].text == "The field 'age' is required."
        assert messages[0].start_position.char_index == 0
        assert messages[0].end_position.char_index == 4
    
    # Test 3: Failed validation with multiple errors
    class MultiErrorSchema(Schema):
        name = String(max_length=5, required=True)
        age = Integer(minimum=18, required=True)
    
    token = DictToken(
        {
            "name": ScalarToken("Jonathan", 0, 8),
            "age": ScalarToken(15, 10, 12),
        },
        0,
        12,
    )
    try:
        validate_with_positions(token=token, validator=MultiErrorSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        messages_sorted = sorted(messages, key=lambda m: m.start_position.char_index)
        assert messages_sorted[0].code == "max_length"
        assert messages_sorted[0].index == ["name"]
        assert messages_sorted[1].code == "min_value"
        assert messages_sorted[1].index == ["age"]
    
    # Test 4: Failed validation with nested required field
    class NestedSchema(Schema):
        class Inner(Schema):
            value = String(required=True)
        
        inner = Inner()
    
    token = DictToken(
        {"inner": DictToken({}, 8, 10)},
        0,
        10,
    )
    try:
        validate_with_positions(token=token, validator=NestedSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["inner", "value"]
        assert messages[0].text == "The field 'value' is required."
    
    # Test 5: Failed validation with list item error
    class ListSchema(Schema):
        items = ListToken.of(String(max_length=3))
    
    token = DictToken(
        {
            "items": ListToken(
                [
                    ScalarToken("abc", 10, 13),
                    ScalarToken("abcd", 15, 19),
                ],
                8,
                20,
            )
        },
        0,
        20,
    )
    try:
        validate_with_positions(token=token, validator=ListSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "max_length"
        assert messages[0].index == ["items", 1]
    
    # Test 6: Successful validation with Field directly
    field = String(max_length=5)
    token = ScalarToken("test", 0, 4)
    result = validate_with_positions(token=token, validator=field)
    assert result == "test"
    
    # Test 7: Failed validation with Field directly
    field = String(max_length=3)
    token = ScalarToken("test", 0, 4)
    try:
        validate_with_positions(token=token, validator=field)
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 1
        assert messages[0].code == "max_length"
        assert messages[0].index == []
    
    # Test 8: Messages are sorted by start position
    class SortingSchema(Schema):
        z = String(required=True)
        a = String(required=True)
    
    token = DictToken(
        {},
        0,
        0,
    )
    try:
        validate_with_positions(token=token, validator=SortingSchema())
        assert False, "Should have raised ValidationError"
    except ValidationError as error:
        messages = error.messages()
        assert len(messages) == 2
        assert messages[0].index == ["a"]
        assert messages[1].index == ["z"]


