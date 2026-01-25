####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="error_code")
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="error_code")
    msg2 = Message(text="Error 2", code="error_code")
    assert msg1 != msg2


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="error_code", index=["field1"])
    msg2 = Message(text="Error", code="error_code", index=["field2"])
    assert msg1 != msg2


def test_message_eq_with_key_vs_index():
    msg1 = Message(text="Error", code="error_code", key="username")
    msg2 = Message(text="Error", code="error_code", index=["username"])
    assert msg1 == msg2


def test_message_eq_with_position():
    pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="error_code", position=pos)
    msg2 = Message(text="Error", code="error_code", position=pos)
    assert msg1 == msg2


def test_message_eq_with_different_positions():
    pos1 = Position(line=1, column=5)
    pos2 = Position(line=2, column=10)
    msg1 = Message(text="Error", code="error_code", position=pos1)
    msg2 = Message(text="Error", code="error_code", position=pos2)
    assert msg1 != msg2


def test_message_eq_with_start_end_positions():
    start_pos = Position(line=1, column=0)
    end_pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="error_code", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="error_code", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2


def test_message_eq_position_vs_start_end():
    pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="error_code", position=pos)
    msg2 = Message(text="Error", code="error_code", start_position=pos, end_position=pos)
    assert msg1 == msg2


def test_message_eq_none_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code=None)
    assert msg1 == msg2


def test_message_eq_nested_index():
    msg1 = Message(text="Error", code="error_code", index=["users", 0, "email"])
    msg2 = Message(text="Error", code="error_code", index=["users", 0, "email"])
    assert msg1 == msg2


def test_message_eq_not_message_instance():
    msg = Message(text="Error", code="error_code")
    assert msg != "not a message"
    assert msg != 42
    assert msg != None
    assert msg != {"text": "Error", "code": "error_code"}


# LLM-generated content at query #2
#--------------------------

```python
def test_position_eq_same_values():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 5, 10)
    assert pos1 == pos2


def test_position_eq_different_line_no():
    pos1 = Position(1, 5, 10)
    pos2 = Position(2, 5, 10)
    assert not (pos1 == pos2)


def test_position_eq_different_column_no():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 6, 10)
    assert not (pos1 == pos2)


def test_position_eq_different_char_index():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 5, 11)
    assert not (pos1 == pos2)


def test_position_eq_all_different():
    pos1 = Position(1, 5, 10)
    pos2 = Position(2, 6, 11)
    assert not (pos1 == pos2)


def test_position_eq_with_non_position_object():
    pos1 = Position(1, 5, 10)
    assert not (pos1 == "not a position")


def test_position_eq_with_none():
    pos1 = Position(1, 5, 10)
    assert not (pos1 == None)


def test_position_eq_with_dict():
    pos1 = Position(1, 5, 10)
    assert not (pos1 == {"line_no": 1, "column_no": 5, "char_index": 10})


def test_position_eq_with_zero_values():
    pos1 = Position(0, 0, 0)
    pos2 = Position(0, 0, 0)
    assert pos1 == pos2


def test_position_eq_same_instance():
    pos1 = Position(1, 5, 10)
    assert pos1 == pos1


# LLM-generated content at query #3
#--------------------------

```python
def test_validation_result_iter():
    from typesystem import ValidationResult, ValidationError
    
    # Test __iter__ with value
    result_with_value = ValidationResult(value="test_data")
    value, error = result_with_value
    assert value == "test_data"
    assert error is None
    
    # Test __iter__ with error
    validation_error = ValidationError(text="Test error")
    result_with_error = ValidationResult(error=validation_error)
    value, error = result_with_error
    assert value is None
    assert error == validation_error
    
    # Test __iter__ returns an iterator
    result = ValidationResult(value=42)
    iterator = iter(result)
    assert next(iterator) == 42
    assert next(iterator) is None


# LLM-generated content at query #4
#--------------------------

```python
def test_repr_basic():
    message = Message(text="Error message", code="error_code")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


def test_repr_with_index():
    message = Message(text="Error message", code="error_code", index=["field", 0, "name"])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['field', 0, 'name'])"


def test_repr_with_key():
    message = Message(text="Error message", code="error_code", key="username")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['username'])"


def test_repr_with_position():
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', position={pos!r})"


def test_repr_with_start_and_end_position():
    start_pos = Position(line=1, column=5)
    end_pos = Position(line=1, column=15)
    message = Message(text="Error message", code="error_code", start_position=start_pos, end_position=end_pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', start_position={start_pos!r}, end_position={end_pos!r})"


def test_repr_with_index_and_position():
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", index=["field"], position=pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', index=['field'], position={pos!r})"


def test_repr_with_index_and_different_positions():
    start_pos = Position(line=1, column=5)
    end_pos = Position(line=2, column=10)
    message = Message(text="Error message", code="error_code", index=["field"], start_position=start_pos, end_position=end_pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', index=['field'], start_position={start_pos!r}, end_position={end_pos!r})"


def test_repr_default_code():
    message = Message(text="Error message")
    result = repr(message)
    assert result == "Message(text='Error message', code='custom')"


def test_repr_empty_index():
    message = Message(text="Error message", code="error_code", index=[])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


def test_repr_with_special_characters():
    message = Message(text="Error with 'quotes' and \"double quotes\"", code="special")
    result = repr(message)
    assert "Error with 'quotes' and \"double quotes\"" in result
    assert "code='special'" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_baseError_constructor_with_single_message():
    error = BaseError(text="Invalid input", code="invalid", key="username")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Invalid input"
    assert error._messages[0].code == "invalid"
    assert error._messages[0].index == ["username"]


def test_baseError_constructor_with_text_only():
    error = BaseError(text="Error message")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == []


def test_baseError_constructor_with_position():
    pos = Position(line_no=1, column_no=5, char_index=4)
    error = BaseError(text="Error at position", code="position_error", position=pos)
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error at position"
    assert error._messages[0].start_position == pos
    assert error._messages[0].end_position == pos


def test_baseError_constructor_with_messages_list():
    msg1 = Message(text="First error", code="error1", key="field1")
    msg2 = Message(text="Second error", code="error2", key="field2")
    error = BaseError(messages=[msg1, msg2])
    assert len(error._messages) == 2
    assert error._messages[0] == msg1
    assert error._messages[1] == msg2


def test_baseError_constructor_populates_message_dict():
    msg = Message(text="Username required", code="required", key="username")
    error = BaseError(messages=[msg])
    assert error._message_dict["username"] == "Username required"


def test_baseError_constructor_populates_nested_message_dict():
    msg = Message(text="Invalid email", code="invalid_email", index=["users", 0, "email"])
    error = BaseError(messages=[msg])
    assert error._message_dict["users"][0]["email"] == "Invalid email"


def test_baseError_constructor_with_empty_index():
    msg = Message(text="General error", code="general")
    error = BaseError(messages=[msg])
    assert error._message_dict[""] == "General error"


def test_baseError_constructor_message_dict_multiple_messages():
    msg1 = Message(text="Error 1", code="code1", key="field1")
    msg2 = Message(text="Error 2", code="code2", key="field2")
    error = BaseError(messages=[msg1, msg2])
    assert error._message_dict["field1"] == "Error 1"
    assert error._message_dict["field2"] == "Error 2"


# LLM-generated content at query #6
#--------------------------

```python
def test_repr_basic():
    message = Message(text="Error message", code="error_code")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


def test_repr_with_index():
    message = Message(text="Error message", code="error_code", index=["field", 0])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['field', 0])"


def test_repr_with_key():
    message = Message(text="Error message", code="error_code", key="username")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['username'])"


def test_repr_with_single_position():
    from typesystem.validation import Position
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', position={pos!r})"


def test_repr_with_start_and_end_position():
    from typesystem.validation import Position
    start_pos = Position(line=1, column=5)
    end_pos = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_pos, end_position=end_pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', start_position={start_pos!r}, end_position={end_pos!r})"


def test_repr_with_index_and_position():
    from typesystem.validation import Position
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", index=["users", 2], position=pos)
    result = repr(message)
    assert result == f"Message(text='Error message', code='error_code', index=['users', 2], position={pos!r})"


def test_repr_default_code():
    message = Message(text="Error message")
    result = repr(message)
    assert result == "Message(text='Error message', code='custom')"


def test_repr_empty_index():
    message = Message(text="Error message", code="error_code", index=[])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    result = message1 == message2
    assert result is False


# LLM-generated content at query #8
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="test_code", key="field1")
    msg2 = Message(text="Error", code="test_code", key="field1")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="test_code", key="field1")
    msg2 = Message(text="Error 2", code="test_code", key="field1")
    assert msg1 != msg2


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1", key="field1")
    msg2 = Message(text="Error", code="code2", key="field1")
    assert msg1 != msg2


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="test_code", key="field1")
    msg2 = Message(text="Error", code="test_code", key="field2")
    assert msg1 != msg2


def test_message_eq_different_start_position():
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=1, column=5)
    msg1 = Message(text="Error", code="test_code", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="test_code", start_position=pos2, end_position=pos2)
    assert msg1 != msg2


def test_message_eq_different_end_position():
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=1, column=5)
    msg1 = Message(text="Error", code="test_code", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="test_code", start_position=pos1, end_position=pos2)
    assert msg1 != msg2


def test_message_eq_with_position_parameter():
    pos = Position(line=1, column=0)
    msg1 = Message(text="Error", code="test_code", position=pos)
    msg2 = Message(text="Error", code="test_code", position=pos)
    assert msg1 == msg2


def test_message_eq_default_code():
    msg1 = Message(text="Error", key="field1")
    msg2 = Message(text="Error", key="field1")
    assert msg1 == msg2


def test_message_eq_with_index_list():
    msg1 = Message(text="Error", code="test_code", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="test_code", index=["users", 0, "name"])
    assert msg1 == msg2


def test_message_eq_different_index_list():
    msg1 = Message(text="Error", code="test_code", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="test_code", index=["users", 1, "name"])
    assert msg1 != msg2


def test_message_eq_not_message_instance():
    msg = Message(text="Error", code="test_code", key="field1")
    assert msg != "Not a message"


def test_message_eq_not_message_instance_dict():
    msg = Message(text="Error", code="test_code", key="field1")
    assert msg != {"text": "Error", "code": "test_code"}


def test_message_eq_none():
    msg = Message(text="Error", code="test_code", key="field1")
    assert msg != None


def test_message_eq_with_none_positions():
    msg1 = Message(text="Error", code="test_code")
    msg2 = Message(text="Error", code="test_code")
    assert msg1 == msg2


# LLM-generated content at query #9
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="code1")
    message2 = Message(text="Error 2", code="code1")
    assert message1 != message2


# LLM-generated content at query #10
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="test_code")
    msg2 = Message(text="Error", code="test_code")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="test_code")
    msg2 = Message(text="Error 2", code="test_code")
    assert not (msg1 == msg2)


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="test_code", index=["field1"])
    msg2 = Message(text="Error", code="test_code", index=["field2"])
    assert not (msg1 == msg2)


def test_message_eq_different_start_position():
    from typesystem.positions import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=1)
    msg1 = Message(text="Error", code="test_code", start_position=pos1)
    msg2 = Message(text="Error", code="test_code", start_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_different_end_position():
    from typesystem.positions import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error", code="test_code", end_position=pos1)
    msg2 = Message(text="Error", code="test_code", end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_with_position():
    from typesystem.positions import Position
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", code="test_code", position=pos)
    msg2 = Message(text="Error", code="test_code", position=pos)
    assert msg1 == msg2


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="test_code", key="username")
    msg2 = Message(text="Error", code="test_code", key="username")
    assert msg1 == msg2


def test_message_eq_with_nested_index():
    msg1 = Message(text="Error", code="test_code", index=["users", 0, "email"])
    msg2 = Message(text="Error", code="test_code", index=["users", 0, "email"])
    assert msg1 == msg2


def test_message_eq_not_message_instance():
    msg = Message(text="Error", code="test_code")
    assert not (msg == "not a message")


def test_message_eq_not_message_instance_dict():
    msg = Message(text="Error", code="test_code")
    assert not (msg == {"text": "Error", "code": "test_code"})


def test_message_eq_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


def test_message_eq_empty_index():
    msg1 = Message(text="Error", code="test_code", index=[])
    msg2 = Message(text="Error", code="test_code")
    assert msg1 == msg2


# LLM-generated content at query #11
#--------------------------

```python
def test_repr_predicate_line_6_false():
    from typesystem import Message, Position
    
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=1, column=5)
    
    message = Message(
        text="Error message",
        code="test_code",
        start_position=pos1,
        end_position=pos2
    )
    
    repr_str = repr(message)
    
    assert "start_position=" in repr_str
    assert "end_position=" in repr_str
    assert "position=" not in repr_str


# LLM-generated content at query #12
#--------------------------

```python
def test_repr_predicate_line_6_evaluates_to_false():
    from typesystem import Message, Position
    
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=1, column=5)
    message = Message(text="error", code="test_code", start_position=pos1, end_position=pos2)
    
    repr_str = repr(message)
    
    assert "start_position=" in repr_str
    assert "end_position=" in repr_str
    assert ", position=" not in repr_str


# LLM-generated content at query #13
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="error_code")
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="error_code")
    msg2 = Message(text="Error 2", code="error_code")
    assert not (msg1 == msg2)


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="error_code", index=["field1"])
    msg2 = Message(text="Error", code="error_code", index=["field2"])
    assert not (msg1 == msg2)


def test_message_eq_different_start_position():
    from typesystem import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=1)
    msg1 = Message(text="Error", code="error_code", start_position=pos1)
    msg2 = Message(text="Error", code="error_code", start_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_different_end_position():
    from typesystem import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error", code="error_code", end_position=pos1)
    msg2 = Message(text="Error", code="error_code", end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_with_position():
    from typesystem import Position
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", code="error_code", position=pos)
    msg2 = Message(text="Error", code="error_code", position=pos)
    assert msg1 == msg2


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="error_code", key="username")
    msg2 = Message(text="Error", code="error_code", key="username")
    assert msg1 == msg2


def test_message_eq_with_complex_index():
    msg1 = Message(text="Error", code="error_code", index=["users", 0, "email"])
    msg2 = Message(text="Error", code="error_code", index=["users", 0, "email"])
    assert msg1 == msg2


def test_message_eq_not_equal_to_other_type():
    msg = Message(text="Error", code="error_code")
    assert not (msg == "Error")


def test_message_eq_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


def test_message_eq_empty_index():
    msg1 = Message(text="Error", code="error_code", index=[])
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


def test_message_eq_none_positions():
    msg1 = Message(text="Error", code="error_code", start_position=None, end_position=None)
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


# LLM-generated content at query #14
#--------------------------

```python
def test_repr_with_different_start_and_end_positions():
    from collections import namedtuple
    
    Position = namedtuple('Position', ['line', 'column'])
    
    start_pos = Position(line=1, column=0)
    end_pos = Position(line=1, column=10)
    
    message = Message(
        text="Error message",
        code="test_code",
        start_position=start_pos,
        end_position=end_pos
    )
    
    repr_str = repr(message)
    
    assert "start_position=" in repr_str
    assert "end_position=" in repr_str
    assert ", position=" not in repr_str


# LLM-generated content at query #15
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    assert message1 != message2


# LLM-generated content at query #16
#--------------------------

```python
def test_repr_predicate_line_6_evaluates_to_false():
    from typesystem import Message, Position
    
    position1 = Position(line=1, column=0, char_index=0)
    position2 = Position(line=1, column=5, char_index=5)
    
    message = Message(
        text="Test error",
        code="test_code",
        start_position=position1,
        end_position=position2
    )
    
    repr_str = repr(message)
    
    assert "start_position=" in repr_str
    assert "end_position=" in repr_str
    assert "position=" not in repr_str


# LLM-generated content at query #17
#--------------------------

```python
def test_message_repr_basic():
    message = Message(text="Error message", code="error_code")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


def test_message_repr_with_index():
    message = Message(text="Error message", code="error_code", index=["users", 0, "name"])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['users', 0, 'name'])"


def test_message_repr_with_key():
    message = Message(text="Error message", code="error_code", key="username")
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code', index=['username'])"


def test_message_repr_with_single_position():
    from typesystem import Position
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=pos)
    result = repr(message)
    assert "text='Error message'" in result
    assert "code='error_code'" in result
    assert f"position={pos!r}" in result


def test_message_repr_with_start_and_end_position():
    from typesystem import Position
    start_pos = Position(line=1, column=5)
    end_pos = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_pos, end_position=end_pos)
    result = repr(message)
    assert "text='Error message'" in result
    assert "code='error_code'" in result
    assert f"start_position={start_pos!r}" in result
    assert f"end_position={end_pos!r}" in result


def test_message_repr_with_index_and_position():
    from typesystem import Position
    pos = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", index=["field", 2], position=pos)
    result = repr(message)
    assert "text='Error message'" in result
    assert "code='error_code'" in result
    assert "index=['field', 2]" in result
    assert f"position={pos!r}" in result


def test_message_repr_default_code():
    message = Message(text="Error message")
    result = repr(message)
    assert result == "Message(text='Error message', code='custom')"


def test_message_repr_empty_index():
    message = Message(text="Error message", code="error_code", index=[])
    result = repr(message)
    assert result == "Message(text='Error message', code='error_code')"


# LLM-generated content at query #18
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    assert message1 != message2


# LLM-generated content at query #19
#--------------------------

```python
def test_repr_with_different_start_and_end_positions():
    from typesystem import Message, Position
    
    start_pos = Position(line=1, column=0)
    end_pos = Position(line=1, column=10)
    
    message = Message(
        text="Error message",
        code="test_code",
        start_position=start_pos,
        end_position=end_pos
    )
    
    repr_str = message.__repr__()
    
    assert "start_position=" in repr_str
    assert "end_position=" in repr_str
    assert ", position=" not in repr_str


# LLM-generated content at query #20
#--------------------------

```python
def test_eq_text_mismatch():
    msg1 = Message(text="Error 1", code="error_code")
    msg2 = Message(text="Error 2", code="error_code")
    assert msg1 != msg2


####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_message_eq_same_messages():
    msg1 = Message(text="Error", code="error_code")
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="error_code")
    msg2 = Message(text="Error 2", code="error_code")
    assert not (msg1 == msg2)


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="error_code", index=["field1"])
    msg2 = Message(text="Error", code="error_code", index=["field2"])
    assert not (msg1 == msg2)


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="error_code", key="username")
    msg2 = Message(text="Error", code="error_code", key="username")
    assert msg1 == msg2


def test_message_eq_with_position():
    pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="error_code", position=pos)
    msg2 = Message(text="Error", code="error_code", position=pos)
    assert msg1 == msg2


def test_message_eq_different_start_position():
    pos1 = Position(line=1, column=5)
    pos2 = Position(line=2, column=5)
    msg1 = Message(text="Error", code="error_code", start_position=pos1)
    msg2 = Message(text="Error", code="error_code", start_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_different_end_position():
    pos1 = Position(line=1, column=5)
    pos2 = Position(line=1, column=10)
    msg1 = Message(text="Error", code="error_code", end_position=pos1)
    msg2 = Message(text="Error", code="error_code", end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_with_not_message_object():
    msg = Message(text="Error", code="error_code")
    assert not (msg == "not a message")


def test_message_eq_with_none():
    msg = Message(text="Error", code="error_code")
    assert not (msg == None)


def test_message_eq_complex_index():
    msg1 = Message(text="Error", code="error_code", index=["users", 3, "username"])
    msg2 = Message(text="Error", code="error_code", index=["users", 3, "username"])
    assert msg1 == msg2


def test_message_eq_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


def test_message_eq_empty_index():
    msg1 = Message(text="Error", code="error_code")
    msg2 = Message(text="Error", code="error_code", index=[])
    assert msg1 == msg2


# LLM-generated content at query #2
#--------------------------

```python
def test_position_eq_identical_positions():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 5, 10)
    assert pos1 == pos2


def test_position_eq_different_line_no():
    pos1 = Position(1, 5, 10)
    pos2 = Position(2, 5, 10)
    assert not (pos1 == pos2)


def test_position_eq_different_column_no():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 6, 10)
    assert not (pos1 == pos2)


def test_position_eq_different_char_index():
    pos1 = Position(1, 5, 10)
    pos2 = Position(1, 5, 11)
    assert not (pos1 == pos2)


def test_position_eq_all_different():
    pos1 = Position(1, 5, 10)
    pos2 = Position(2, 6, 11)
    assert not (pos1 == pos2)


def test_position_eq_with_non_position_object():
    pos = Position(1, 5, 10)
    assert not (pos == "not a position")


def test_position_eq_with_none():
    pos = Position(1, 5, 10)
    assert not (pos == None)


def test_position_eq_with_dict():
    pos = Position(1, 5, 10)
    assert not (pos == {"line_no": 1, "column_no": 5, "char_index": 10})


def test_position_eq_zero_values():
    pos1 = Position(0, 0, 0)
    pos2 = Position(0, 0, 0)
    assert pos1 == pos2


def test_position_eq_negative_values():
    pos1 = Position(-1, -5, -10)
    pos2 = Position(-1, -5, -10)
    assert pos1 == pos2


# LLM-generated content at query #3
#--------------------------

```python
def test_baseError_constructor_with_single_message():
    error = BaseError(text="Invalid input", code="invalid", key="username")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Invalid input"
    assert error._messages[0].code == "invalid"
    assert error._messages[0].index == ["username"]


def test_baseError_constructor_with_text_only():
    error = BaseError(text="Error message")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == []


def test_baseError_constructor_with_position():
    pos = Position(line_no=1, column_no=5, char_index=4)
    error = BaseError(text="Error at position", code="syntax_error", position=pos)
    assert error._messages[0].text == "Error at position"
    assert error._messages[0].start_position == pos
    assert error._messages[0].end_position == pos


def test_baseError_constructor_with_multiple_messages():
    msg1 = Message(text="Error 1", code="code1", key="field1")
    msg2 = Message(text="Error 2", code="code2", key="field2")
    error = BaseError(messages=[msg1, msg2])
    assert len(error._messages) == 2
    assert error._messages[0] == msg1
    assert error._messages[1] == msg2


def test_baseError_constructor_message_dict_single_message():
    error = BaseError(text="Invalid username", code="invalid_format", key="username")
    assert error._message_dict == {"username": "Invalid username"}


def test_baseError_constructor_message_dict_no_key():
    error = BaseError(text="Generic error")
    assert error._message_dict == {"": "Generic error"}


def test_baseError_constructor_message_dict_nested():
    msg1 = Message(text="Too short", code="min_length", index=["users", 0, "name"])
    msg2 = Message(text="Invalid email", code="invalid_email", index=["users", 0, "email"])
    error = BaseError(messages=[msg1, msg2])
    assert error._message_dict == {"users": {0: {"name": "Too short", "email": "Invalid email"}}}


def test_baseError_constructor_empty_messages_list_assertion():
    try:
        error = BaseError(messages=[])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_baseError_constructor_conflicting_text_and_messages_assertion():
    try:
        msg = Message(text="Error", code="code1")
        error = BaseError(text="Should fail", messages=[msg])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_baseError_constructor_conflicting_code_and_messages_assertion():
    try:
        msg = Message(text="Error", code="code1")
        error = BaseError(code="should_fail", messages=[msg])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_baseError_constructor_conflicting_key_and_messages_assertion():
    try:
        msg = Message(text="Error", code="code1")
        error = BaseError(key="should_fail", messages=[msg])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_baseError_constructor_conflicting_position_and_messages_assertion():
    try:
        pos = Position(line_no=1, column_no=5, char_index=4)
        msg = Message(text="Error", code="code1")
        error = BaseError(position=pos, messages=[msg])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_validation_result_repr_with_error():
    from typesystem import ValidationResult, ValidationError
    
    error = ValidationError(text="Invalid input")
    result = ValidationResult(error=error)
    repr_str = repr(result)
    
    assert "ValidationResult" in repr_str
    assert "error=" in repr_str
    assert "Invalid input" in repr_str


def test_validation_result_repr_with_value():
    from typesystem import ValidationResult
    
    result = ValidationResult(value="test_data")
    repr_str = repr(result)
    
    assert "ValidationResult" in repr_str
    assert "value=" in repr_str
    assert "test_data" in repr_str


def test_validation_result_repr_with_none_value():
    from typesystem import ValidationResult
    
    result = ValidationResult(value=None)
    repr_str = repr(result)
    
    assert "ValidationResult" in repr_str
    assert "value=" in repr_str
    assert "None" in repr_str


def test_validation_result_repr_with_complex_value():
    from typesystem import ValidationResult
    
    complex_value = {"key": "value", "nested": [1, 2, 3]}
    result = ValidationResult(value=complex_value)
    repr_str = repr(result)
    
    assert "ValidationResult" in repr_str
    assert "value=" in repr_str
    assert "key" in repr_str


# LLM-generated content at query #5
#--------------------------

```python
def test_repr_single_message_without_index():
    from typesystem import BaseError, Message
    
    error = BaseError(text="Invalid input", code="invalid")
    result = repr(error)
    assert result == "BaseError(text='Invalid input', code='invalid')"


def test_repr_single_message_with_index():
    from typesystem import BaseError, Message
    
    messages = [Message(text="Invalid input", code="invalid", index=["field"])]
    error = BaseError(messages=messages)
    result = repr(error)
    assert "BaseError([Message" in result


def test_repr_multiple_messages():
    from typesystem import BaseError, Message
    
    messages = [
        Message(text="Error 1", code="error1"),
        Message(text="Error 2", code="error2")
    ]
    error = BaseError(messages=messages)
    result = repr(error)
    assert "BaseError([Message" in result
    assert "Error 1" in result
    assert "Error 2" in result


def test_repr_single_message_no_code():
    from typesystem import BaseError
    
    error = BaseError(text="Just a message")
    result = repr(error)
    assert result == "BaseError(text='Just a message', code=None)"


def test_repr_nested_messages():
    from typesystem import BaseError, Message
    
    messages = [
        Message(text="Nested error", code="nested", index=["parent", "child"])
    ]
    error = BaseError(messages=messages)
    result = repr(error)
    assert "BaseError([Message" in result
    assert "Nested error" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="test_code")
    msg2 = Message(text="Error", code="test_code")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="test_code")
    msg2 = Message(text="Error 2", code="test_code")
    assert not (msg1 == msg2)


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="test_code", index=["field1"])
    msg2 = Message(text="Error", code="test_code", index=["field2"])
    assert not (msg1 == msg2)


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="test_code", key="username")
    msg2 = Message(text="Error", code="test_code", key="username")
    assert msg1 == msg2


def test_message_eq_different_start_position():
    from typesystem.message import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=1)
    msg1 = Message(text="Error", code="test_code", start_position=pos1)
    msg2 = Message(text="Error", code="test_code", start_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_different_end_position():
    from typesystem.message import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error", code="test_code", end_position=pos1)
    msg2 = Message(text="Error", code="test_code", end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_with_position_parameter():
    from typesystem.message import Position
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", code="test_code", position=pos)
    msg2 = Message(text="Error", code="test_code", position=pos)
    assert msg1 == msg2


def test_message_eq_with_nested_index():
    msg1 = Message(text="Error", code="test_code", index=["users", 0, "email"])
    msg2 = Message(text="Error", code="test_code", index=["users", 0, "email"])
    assert msg1 == msg2


def test_message_eq_with_non_message_object():
    msg = Message(text="Error", code="test_code")
    assert not (msg == "not a message")


def test_message_eq_with_none():
    msg = Message(text="Error", code="test_code")
    assert not (msg == None)


def test_message_eq_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2


def test_message_eq_empty_index():
    msg1 = Message(text="Error", code="test_code", index=[])
    msg2 = Message(text="Error", code="test_code", index=[])
    assert msg1 == msg2


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    assert message1 != message2


# LLM-generated content at query #8
#--------------------------

```python
def test_message_eq_same_attributes():
    msg1 = Message(text="Error", code="error_code", index=[0, "field"])
    msg2 = Message(text="Error", code="error_code", index=[0, "field"])
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="error_code")
    msg2 = Message(text="Error 2", code="error_code")
    assert msg1 != msg2


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="error_code", index=[0])
    msg2 = Message(text="Error", code="error_code", index=[1])
    assert msg1 != msg2


def test_message_eq_different_start_position():
    from typesystem import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=1)
    msg1 = Message(text="Error", code="error_code", start_position=pos1)
    msg2 = Message(text="Error", code="error_code", start_position=pos2)
    assert msg1 != msg2


def test_message_eq_different_end_position():
    from typesystem import Position
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error", code="error_code", end_position=pos1)
    msg2 = Message(text="Error", code="error_code", end_position=pos2)
    assert msg1 != msg2


def test_message_eq_with_position():
    from typesystem import Position
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", code="error_code", position=pos)
    msg2 = Message(text="Error", code="error_code", position=pos)
    assert msg1 == msg2


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="error_code", key="username")
    msg2 = Message(text="Error", code="error_code", key="username")
    assert msg1 == msg2


def test_message_eq_with_none_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2


def test_message_eq_not_message_instance():
    msg = Message(text="Error", code="error_code")
    assert msg != "not a message"
    assert msg != 42
    assert msg != None
    assert msg != {"text": "Error", "code": "error_code"}


def test_message_eq_empty_index():
    msg1 = Message(text="Error", code="error_code")
    msg2 = Message(text="Error", code="error_code")
    assert msg1 == msg2


def test_message_eq_complex_index():
    msg1 = Message(text="Error", code="error_code", index=["users", 0, "email", 1])
    msg2 = Message(text="Error", code="error_code", index=["users", 0, "email", 1])
    assert msg1 == msg2


# LLM-generated content at query #9
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="error_code")
    message2 = Message(text="Error 2", code="error_code")
    assert message1 != message2


# LLM-generated content at query #10
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    assert message1 != message2


# LLM-generated content at query #11
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="error_code")
    message2 = Message(text="Error 2", code="error_code")
    assert message1 != message2


# LLM-generated content at query #12
#--------------------------

```python
def test_eq_different_text():
    message1 = Message(text="Error 1", code="test_code")
    message2 = Message(text="Error 2", code="test_code")
    assert message1 != message2


# LLM-generated content at query #13
#--------------------------

```python
def test_message_eq_identical_messages():
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


def test_message_eq_different_text():
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert not (msg1 == msg2)


def test_message_eq_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)


def test_message_eq_different_index():
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert not (msg1 == msg2)


def test_message_eq_different_start_position():
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=2, column=0)
    msg1 = Message(text="Error", code="custom", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="custom", start_position=pos2, end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_different_end_position():
    pos1 = Position(line=1, column=0)
    pos2 = Position(line=1, column=5)
    msg1 = Message(text="Error", code="custom", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="custom", start_position=pos1, end_position=pos2)
    assert not (msg1 == msg2)


def test_message_eq_with_key():
    msg1 = Message(text="Error", code="custom", key="username")
    msg2 = Message(text="Error", code="custom", key="username")
    assert msg1 == msg2


def test_message_eq_with_index():
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2


def test_message_eq_with_position():
    pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", position=pos)
    assert msg1 == msg2


def test_message_eq_not_message_instance():
    msg = Message(text="Error", code="custom")
    assert not (msg == "Not a message")


def test_message_eq_not_message_instance_dict():
    msg = Message(text="Error", code="custom")
    assert not (msg == {"text": "Error", "code": "custom"})


def test_message_eq_all_fields_match():
    pos = Position(line=1, column=0)
    msg1 = Message(
        text="Error message",
        code="max_length",
        index=["field", 0],
        start_position=pos,
        end_position=pos
    )
    msg2 = Message(
        text="Error message",
        code="max_length",
        index=["field", 0],
        start_position=pos,
        end_position=pos
    )
    assert msg1 == msg2


def test_message_eq_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


