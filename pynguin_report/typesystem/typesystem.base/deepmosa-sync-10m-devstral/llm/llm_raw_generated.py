####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    assert msg1 == msg2

def test_message_equality_with_different_text():
    msg1 = Message(text="Error1", code="test", key="field")
    msg2 = Message(text="Error2", code="test", key="field")
    assert not (msg1 == msg2)

def test_message_equality_with_different_code():
    msg1 = Message(text="Error", code="test1", key="field")
    msg2 = Message(text="Error", code="test2", key="field")
    assert not (msg1 == msg2)

def test_message_equality_with_different_index():
    msg1 = Message(text="Error", code="test", index=["a", 1])
    msg2 = Message(text="Error", code="test", index=["b", 2])
    assert not (msg1 == msg2)

def test_message_equality_with_different_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="test", position=pos1)
    msg2 = Message(text="Error", code="test", position=pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_none_vs_custom_code():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2

def test_message_equality_with_start_end_positions():
    start_pos = Position(line=1, column=1)
    end_pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2

def test_message_equality_with_different_start_position():
    start_pos1 = Position(line=1, column=1)
    start_pos2 = Position(line=2, column=1)
    end_pos = Position(line=1, column=5)
    msg1 = Message(text="Error", code="test", start_position=start_pos1, end_position=end_pos)
    msg2 = Message(text="Error", code="test", start_position=start_pos2, end_position=end_pos)
    assert not (msg1 == msg2)

def test_message_equality_with_different_end_position():
    start_pos = Position(line=1, column=1)
    end_pos1 = Position(line=1, column=5)
    end_pos2 = Position(line=1, column=10)
    msg1 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos1)
    msg2 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_none_vs_position():
    msg1 = Message(text="Error", code="test", start_position=None, end_position=None)
    msg2 = Message(text="Error", code="test", position=None)
    assert msg1 == msg2

def test_message_equality_with_non_message_object():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == "not a message")


# LLM-generated content at query #2
#--------------------------

```python
def test___repr___single_message_no_index():
    error = BaseError(text="Invalid value", code="invalid")
    assert repr(error) == "BaseError(text='Invalid value', code='invalid')"

def test___repr___single_message_with_index():
    error = BaseError(text="Invalid value", code="invalid", key="field")
    assert repr(error) == "BaseError([Message(text='Invalid value', code='invalid', index=['field'])]"

def test___repr___multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", index=["key1"]),
        Message(text="Error 2", code="code2", index=["key2"])
    ]
    error = BaseError(messages=messages)
    assert repr(error) == "BaseError([Message(text='Error 1', code='code1', index=['key1']), Message(text='Error 2', code='code2', index=['key2'])]"


# LLM-generated content at query #3
#--------------------------

```python
def test_validation_result_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_validation_result_iter_with_error():
    error = ValidationError("test error")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error

def test_validation_result_iter_returns_iterator():
    result = ValidationResult(value=42)
    iterator = iter(result)
    assert hasattr(iterator, '__next__')


# LLM-generated content at query #4
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


# LLM-generated content at query #5
#--------------------------

```python
def test___str___single_message_no_index():
    error = BaseError(text="Error message", code="error_code")
    assert str(error) == "Error message"

def test___str___multiple_messages():
    messages = [
        Message(text="Error 1", code="error1", index=[0]),
        Message(text="Error 2", code="error2", index=[1])
    ]
    error = BaseError(messages=messages)
    assert str(error) == "{0: 'Error 1', 1: 'Error 2'}"


# LLM-generated content at query #6
#--------------------------

```python
def test_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


# LLM-generated content at query #7
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1", code="error", key="field")
    msg2 = Message(text="Error 2", code="error", key="field")
    assert not (msg1 == msg2)


# LLM-generated content at query #8
#--------------------------

```python
def test_validation_result_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_validation_result_iter_with_error():
    error = ValidationError("Invalid data")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error

def test_validation_result_iter_unpacking():
    result = ValidationResult(value="test")
    items = list(result)
    assert items == ["test", None]


# LLM-generated content at query #9
#--------------------------

```python
def test_eq_with_same_attributes():
    msg1 = Message(text="Error", code="max_length", key="username")
    msg2 = Message(text="Error", code="max_length", key="username")
    assert msg1 == msg2

def test_eq_with_different_text():
    msg1 = Message(text="Error1", code="max_length", key="username")
    msg2 = Message(text="Error2", code="max_length", key="username")
    assert not (msg1 == msg2)

def test_eq_with_different_code():
    msg1 = Message(text="Error", code="max_length", key="username")
    msg2 = Message(text="Error", code="min_length", key="username")
    assert not (msg1 == msg2)

def test_eq_with_different_index():
    msg1 = Message(text="Error", code="max_length", key="username")
    msg2 = Message(text="Error", code="max_length", key="email")
    assert not (msg1 == msg2)

def test_eq_with_different_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="max_length", position=pos1)
    msg2 = Message(text="Error", code="max_length", position=pos2)
    assert not (msg1 == msg2)

def test_eq_with_different_start_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="max_length", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="max_length", start_position=pos2, end_position=pos1)
    assert not (msg1 == msg2)

def test_eq_with_different_end_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="max_length", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="max_length", start_position=pos1, end_position=pos2)
    assert not (msg1 == msg2)

def test_eq_with_non_message_object():
    msg = Message(text="Error", code="max_length", key="username")
    assert not (msg == "not a message")

def test_eq_with_none():
    msg = Message(text="Error", code="max_length", key="username")
    assert not (msg == None)


# LLM-generated content at query #10
#--------------------------

```python
def test_message_equality():
    message1 = Message(text="Error message", code="error", key="field")
    message2 = Message(text="Error message", code="error", key="field")
    assert message1 == message2

def test_message_equality_with_different_text():
    message1 = Message(text="Error message 1", code="error", key="field")
    message2 = Message(text="Error message 2", code="error", key="field")
    assert message1 != message2

def test_message_equality_with_different_code():
    message1 = Message(text="Error message", code="error1", key="field")
    message2 = Message(text="Error message", code="error2", key="field")
    assert message1 != message2

def test_message_equality_with_different_key():
    message1 = Message(text="Error message", code="error", key="field1")
    message2 = Message(text="Error message", code="error", key="field2")
    assert message1 != message2

def test_message_equality_with_different_index():
    message1 = Message(text="Error message", code="error", index=[1, 2])
    message2 = Message(text="Error message", code="error", index=[1, 3])
    assert message1 != message2

def test_message_equality_with_different_position():
    position1 = Position(line=1, column=1)
    position2 = Position(line=2, column=2)
    message1 = Message(text="Error message", code="error", position=position1)
    message2 = Message(text="Error message", code="error", position=position2)
    assert message1 != message2

def test_message_equality_with_different_start_and_end_position():
    start_position1 = Position(line=1, column=1)
    end_position1 = Position(line=1, column=10)
    start_position2 = Position(line=2, column=2)
    end_position2 = Position(line=2, column=20)
    message1 = Message(text="Error message", code="error", start_position=start_position1, end_position=end_position1)
    message2 = Message(text="Error message", code="error", start_position=start_position2, end_position=end_position2)
    assert message1 != message2

def test_message_equality_with_none_position():
    message1 = Message(text="Error message", code="error", key="field")
    message2 = Message(text="Error message", code="error", key="field", start_position=None, end_position=None)
    assert message1 == message2

def test_message_equality_with_different_type():
    message = Message(text="Error message", code="error", key="field")
    assert message != "not a message"


# LLM-generated content at query #11
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


# LLM-generated content at query #12
#--------------------------

```python
def test_eq_same_instance():
    msg = Message(text="Error message")
    assert msg == msg

def test_eq_different_instances_same_attributes():
    msg1 = Message(text="Error message", code="max_length", key="username")
    msg2 = Message(text="Error message", code="max_length", key="username")
    assert msg1 == msg2

def test_eq_different_text():
    msg1 = Message(text="Error message 1")
    msg2 = Message(text="Error message 2")
    assert not (msg1 == msg2)

def test_eq_different_code():
    msg1 = Message(text="Error message", code="max_length")
    msg2 = Message(text="Error message", code="min_length")
    assert not (msg1 == msg2)

def test_eq_different_index():
    msg1 = Message(text="Error message", index=["users", 3, "username"])
    msg2 = Message(text="Error message", index=["users", 4, "username"])
    assert not (msg1 == msg2)

def test_eq_different_start_position():
    from typing import NamedTuple
    class Position(NamedTuple):
        line: int
        column: int
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error message", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error message", start_position=pos2, end_position=pos2)
    assert not (msg1 == msg2)

def test_eq_different_end_position():
    from typing import NamedTuple
    class Position(NamedTuple):
        line: int
        column: int
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=1, column=2)
    msg1 = Message(text="Error message", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error message", start_position=pos1, end_position=pos2)
    assert not (msg1 == msg2)

def test_eq_different_position_vs_start_end():
    from typing import NamedTuple
    class Position(NamedTuple):
        line: int
        column: int
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error message", position=pos)
    msg2 = Message(text="Error message", start_position=pos, end_position=pos)
    assert msg1 == msg2

def test_eq_non_message_object():
    msg = Message(text="Error message")
    assert not (msg == "not a message")

def test_eq_none():
    msg = Message(text="Error message")
    assert not (msg == None)


# LLM-generated content at query #13
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base_error_single_message():
    error = BaseError(text="Error message", code="error_code", key="error_key")
    assert error._messages == [Message(text="Error message", code="error_code", key="error_key")]
    assert error._message_dict == {"error_key": "Error message"}

def test_base_error_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", key="key1"),
        Message(text="Error 2", code="code2", key="key2")
    ]
    error = BaseError(messages=messages)
    assert error._messages == messages
    assert error._message_dict == {"key1": "Error 1", "key2": "Error 2"}

def test_base_error_nested_messages():
    messages = [
        Message(text="Error 1", code="code1", index=["users", 0, "name"]),
        Message(text="Error 2", code="code2", index=["users", 1, "email"])
    ]
    error = BaseError(messages=messages)
    assert error._messages == messages
    assert error._message_dict == {"users": {0: {"name": "Error 1"}, 1: {"email": "Error 2"}}}

def test_base_error_with_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    error = BaseError(text="Error message", code="error_code", position=position)
    assert error._messages == [Message(text="Error message", code="error_code", position=position)]
    assert error._message_dict == {"": "Error message"}


# LLM-generated content at query #2
#--------------------------

```python
def test_message_constructor_with_text_only():
    message = Message(text="Error message")
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_text_and_code():
    message = Message(text="Error message", code="max_length")
    assert message.text == "Error message"
    assert message.code == "max_length"
    assert message.index == []
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_text_and_key():
    message = Message(text="Error message", key="username")
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == ["username"]
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_text_and_index():
    message = Message(text="Error message", index=["users", 3, "username"])
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == ["users", 3, "username"]
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_text_and_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    message = Message(text="Error message", position=position)
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position == position
    assert message.end_position == position

def test_message_constructor_with_text_and_start_end_positions():
    start_position = Position(line_no=1, column_no=2, char_index=3)
    end_position = Position(line_no=4, column_no=5, char_index=6)
    message = Message(text="Error message", start_position=start_position, end_position=end_position)
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position == start_position
    assert message.end_position == end_position

def test_message_constructor_with_all_parameters():
    message = Message(
        text="Error message",
        code="max_length",
        key="username",
        position=Position(line_no=1, column_no=2, char_index=3)
    )
    assert message.text == "Error message"
    assert message.code == "max_length"
    assert message.index == ["username"]
    assert message.start_position == Position(line_no=1, column_no=2, char_index=3)
    assert message.end_position == Position(line_no=1, column_no=2, char_index=3)


# LLM-generated content at query #3
#--------------------------

```python
def test_position_eq_returns_true_for_equal_positions():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 3)
    assert pos1 == pos2

def test_position_eq_returns_false_for_different_line_numbers():
    pos1 = Position(1, 2, 3)
    pos2 = Position(2, 2, 3)
    assert not (pos1 == pos2)

def test_position_eq_returns_false_for_different_column_numbers():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 3, 3)
    assert not (pos1 == pos2)

def test_position_eq_returns_false_for_different_char_indices():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 4)
    assert not (pos1 == pos2)

def test_position_eq_returns_false_for_non_position_objects():
    pos = Position(1, 2, 3)
    assert not (pos == "not a position")


# LLM-generated content at query #4
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    assert msg1 == msg2

def test_message_equality_with_different_text():
    msg1 = Message(text="Error1", code="test", key="field")
    msg2 = Message(text="Error2", code="test", key="field")
    assert not (msg1 == msg2)

def test_message_equality_with_different_code():
    msg1 = Message(text="Error", code="test1", key="field")
    msg2 = Message(text="Error", code="test2", key="field")
    assert not (msg1 == msg2)

def test_message_equality_with_different_index():
    msg1 = Message(text="Error", code="test", key="field1")
    msg2 = Message(text="Error", code="test", key="field2")
    assert not (msg1 == msg2)

def test_message_equality_with_different_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="test", position=pos1)
    msg2 = Message(text="Error", code="test", position=pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_non_message_object():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == "not a message")

def test_message_equality_with_none_position():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    msg1.start_position = None
    msg1.end_position = None
    msg2.start_position = None
    msg2.end_position = None
    assert msg1 == msg2

def test_message_equality_with_start_and_end_position():
    start_pos = Position(line=1, column=1)
    end_pos = Position(line=1, column=10)
    msg1 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2

def test_message_equality_with_different_start_position():
    start_pos1 = Position(line=1, column=1)
    start_pos2 = Position(line=2, column=1)
    end_pos = Position(line=1, column=10)
    msg1 = Message(text="Error", code="test", start_position=start_pos1, end_position=end_pos)
    msg2 = Message(text="Error", code="test", start_position=start_pos2, end_position=end_pos)
    assert not (msg1 == msg2)

def test_message_equality_with_different_end_position():
    start_pos = Position(line=1, column=1)
    end_pos1 = Position(line=1, column=10)
    end_pos2 = Position(line=1, column=20)
    msg1 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos1)
    msg2 = Message(text="Error", code="test", start_position=start_pos, end_position=end_pos2)
    assert not (msg1 == msg2)


# LLM-generated content at query #5
#--------------------------

```python
def test_ValidationResult___iter__():
    result_with_value = ValidationResult(value=42)
    value, error = result_with_value.__iter__()
    assert value == 42
    assert error is None

    result_with_error = ValidationResult(error=ValidationError("error"))
    value, error = result_with_error.__iter__()
    assert value is None
    assert error == ValidationError("error")


# LLM-generated content at query #6
#--------------------------

```python
def test___str___single_message_no_index():
    error = BaseError(text="Invalid value", code="invalid")
    assert str(error) == "Invalid value"

def test___str___single_message_with_index():
    error = BaseError(text="Invalid value", code="invalid", key="field")
    assert str(error) == "{'field': 'Invalid value'}"

def test___str___multiple_messages():
    messages = [
        Message(text="Invalid value", code="invalid", index=["field1"]),
        Message(text="Missing value", code="missing", index=["field2"]),
    ]
    error = BaseError(messages=messages)
    assert str(error) == "{'field1': 'Invalid value', 'field2': 'Missing value'}"


# LLM-generated content at query #7
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    assert msg1 == msg2

def test_message_inequality_with_different_text():
    msg1 = Message(text="Error1", code="test", key="field")
    msg2 = Message(text="Error2", code="test", key="field")
    assert not (msg1 == msg2)

def test_message_inequality_with_different_code():
    msg1 = Message(text="Error", code="test1", key="field")
    msg2 = Message(text="Error", code="test2", key="field")
    assert not (msg1 == msg2)

def test_message_inequality_with_different_index():
    msg1 = Message(text="Error", code="test", index=["a", 1])
    msg2 = Message(text="Error", code="test", index=["b", 2])
    assert not (msg1 == msg2)

def test_message_equality_with_none_code():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", key="field")
    assert msg1 == msg2

def test_message_inequality_with_different_position():
    pos1 = (1, 2)
    pos2 = (3, 4)
    msg1 = Message(text="Error", code="test", position=pos1)
    msg2 = Message(text="Error", code="test", position=pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_start_and_end_position():
    msg1 = Message(text="Error", code="test", start_position=(1, 2), end_position=(3, 4))
    msg2 = Message(text="Error", code="test", start_position=(1, 2), end_position=(3, 4))
    assert msg1 == msg2

def test_message_inequality_with_different_start_position():
    msg1 = Message(text="Error", code="test", start_position=(1, 2), end_position=(3, 4))
    msg2 = Message(text="Error", code="test", start_position=(5, 6), end_position=(3, 4))
    assert not (msg1 == msg2)

def test_message_inequality_with_different_end_position():
    msg1 = Message(text="Error", code="test", start_position=(1, 2), end_position=(3, 4))
    msg2 = Message(text="Error", code="test", start_position=(1, 2), end_position=(5, 6))
    assert not (msg1 == msg2)

def test_message_inequality_with_non_message_object():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == "not a message")

def test_message_equality_with_empty_index():
    msg1 = Message(text="Error", code="test")
    msg2 = Message(text="Error", code="test")
    assert msg1 == msg2


# LLM-generated content at query #8
#--------------------------

```python
def test_validation_result_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_validation_result_iter_with_error():
    error = ValidationError("Invalid data")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error


# LLM-generated content at query #9
#--------------------------

```python
def test_repr_single_message_no_index():
    error = BaseError(text="Invalid value", code="invalid")
    assert repr(error) == "BaseError(text='Invalid value', code='invalid')"

def test_repr_single_message_with_index():
    error = BaseError(text="Invalid value", code="invalid", key="field")
    assert repr(error) == "BaseError([Message(text='Invalid value', code='invalid', index=['field'])]"

def test_repr_multiple_messages():
    messages = [
        Message(text="Invalid value", code="invalid", index=["field1"]),
        Message(text="Missing value", code="missing", index=["field2"])
    ]
    error = BaseError(messages=messages)
    assert repr(error) == "BaseError([Message(text='Invalid value', code='invalid', index=['field1']), Message(text='Missing value', code='missing', index=['field2'])]"


# LLM-generated content at query #10
#--------------------------

```python
def test_message_repr_with_text_and_code():
    message = Message(text="Error message", code="error_code")
    assert repr(message) == "Message(text='Error message', code='error_code')"

def test_message_repr_with_key():
    message = Message(text="Error message", code="error_code", key="username")
    assert repr(message) == "Message(text='Error message', code='error_code', index=['username'])"

def test_message_repr_with_index():
    message = Message(text="Error message", code="error_code", index=["users", 3, "username"])
    assert repr(message) == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_message_repr_with_position():
    position = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=position)
    assert repr(message) == f"Message(text='Error message', code='error_code', position={position!r})"

def test_message_repr_with_start_and_end_position():
    start_position = Position(line=1, column=5)
    end_position = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_position, end_position=end_position)
    assert repr(message) == f"Message(text='Error message', code='error_code', start_position={start_position!r}, end_position={end_position!r})"

def test_message_repr_with_default_code():
    message = Message(text="Error message")
    assert repr(message) == "Message(text='Error message', code='custom')"


# LLM-generated content at query #11
#--------------------------

```python
def test_message_repr_with_text_and_code():
    message = Message(text="Error message", code="error_code")
    assert repr(message) == "Message(text='Error message', code='error_code')"

def test_message_repr_with_text_code_and_key():
    message = Message(text="Error message", code="error_code", key="username")
    assert repr(message) == "Message(text='Error message', code='error_code', index=['username'])"

def test_message_repr_with_text_code_and_index():
    message = Message(text="Error message", code="error_code", index=["users", 3, "username"])
    assert repr(message) == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_message_repr_with_text_code_and_position():
    position = Position(line=1, column=1)
    message = Message(text="Error message", code="error_code", position=position)
    assert repr(message) == f"Message(text='Error message', code='error_code', position={position!r})"

def test_message_repr_with_text_code_start_and_end_position():
    start_position = Position(line=1, column=1)
    end_position = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_position, end_position=end_position)
    assert repr(message) == f"Message(text='Error message', code='error_code', start_position={start_position!r}, end_position={end_position!r})"


# LLM-generated content at query #12
#--------------------------

```python
def test_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_iter_with_error():
    error = ValidationError("test error")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error


# LLM-generated content at query #13
#--------------------------

```python
def test__iter__returns_iterator_with_value_and_error():
    value = "test_value"
    error = ValidationError("test_error")
    result = ValidationResult(value=value, error=error)
    iterator = iter(result)
    assert next(iterator) == value
    assert next(iterator) == error


# LLM-generated content at query #14
#--------------------------

```python
def test_start_position_not_equal_to_end_position():
    message = Message(
        text="Error",
        start_position=Position(line=1, column=1),
        end_position=Position(line=2, column=2)
    )
    assert message.start_position != message.end_position


# LLM-generated content at query #15
#--------------------------

```python
def test_start_position_not_equal_to_end_position():
    message = Message(
        text="Error message",
        start_position=Position(line=1, column=1),
        end_position=Position(line=2, column=1)
    )
    assert message.start_position != message.end_position


# LLM-generated content at query #16
#--------------------------

```python
def test__iter__returns_iterator_with_value_and_error():
    value = "test_value"
    error = ValidationError("test_error")
    result = ValidationResult(value=value, error=error)
    iterator = iter(result)
    assert next(iterator) == value
    assert next(iterator) == error


# LLM-generated content at query #17
#--------------------------

```python
def test_message_repr_with_text_and_code():
    message = Message(text="Error message", code="error_code")
    assert repr(message) == "Message(text='Error message', code='error_code')"

def test_message_repr_with_key():
    message = Message(text="Error message", key="username")
    assert repr(message) == "Message(text='Error message', code='custom', index=['username'])"

def test_message_repr_with_index():
    message = Message(text="Error message", index=["users", 3, "username"])
    assert repr(message) == "Message(text='Error message', code='custom', index=['users', 3, 'username'])"

def test_message_repr_with_position():
    position = Position(line=1, column=5)
    message = Message(text="Error message", position=position)
    assert repr(message) == f"Message(text='Error message', code='custom', position={position!r})"

def test_message_repr_with_start_and_end_position():
    start_position = Position(line=1, column=5)
    end_position = Position(line=1, column=10)
    message = Message(text="Error message", start_position=start_position, end_position=end_position)
    assert repr(message) == f"Message(text='Error message', code='custom', start_position={start_position!r}, end_position={end_position!r})"


# LLM-generated content at query #18
#--------------------------

```python
def test_repr_with_text_and_code():
    message = Message(text="Error message", code="error_code")
    assert repr(message) == "Message(text='Error message', code='error_code')"

def test_repr_with_text_code_and_key():
    message = Message(text="Error message", code="error_code", key="username")
    assert repr(message) == "Message(text='Error message', code='error_code', index=['username'])"

def test_repr_with_text_code_and_index():
    message = Message(text="Error message", code="error_code", index=["users", 3, "username"])
    assert repr(message) == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_repr_with_text_code_and_position():
    position = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=position)
    assert repr(message) == f"Message(text='Error message', code='error_code', position={repr(position)})"

def test_repr_with_text_code_start_and_end_position():
    start_position = Position(line=1, column=5)
    end_position = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_position, end_position=end_position)
    assert repr(message) == f"Message(text='Error message', code='error_code', start_position={repr(start_position)}, end_position={repr(end_position)})"

def test_repr_with_default_code():
    message = Message(text="Error message")
    assert repr(message) == "Message(text='Error message', code='custom')"


# LLM-generated content at query #19
#--------------------------

```python
def test_ValidationResult___iter___with_value():
    result = ValidationResult(value=42)
    value, error = tuple(result)
    assert value == 42
    assert error is None

def test_ValidationResult___iter___with_error():
    error = ValidationError("test error")
    result = ValidationResult(error=error)
    value, error = tuple(result)
    assert value is None
    assert error == error


# LLM-generated content at query #20
#--------------------------

```python
def test_message_repr_with_text_and_code():
    message = Message(text="Error message", code="error_code")
    assert repr(message) == "Message(text='Error message', code='error_code')"

def test_message_repr_with_text_code_and_key():
    message = Message(text="Error message", code="error_code", key="username")
    assert repr(message) == "Message(text='Error message', code='error_code', index=['username'])"

def test_message_repr_with_text_code_and_index():
    message = Message(text="Error message", code="error_code", index=["users", 3, "username"])
    assert repr(message) == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_message_repr_with_text_code_and_position():
    position = Position(line=1, column=5)
    message = Message(text="Error message", code="error_code", position=position)
    assert repr(message) == f"Message(text='Error message', code='error_code', position={position!r})"

def test_message_repr_with_text_code_and_start_end_positions():
    start_position = Position(line=1, column=5)
    end_position = Position(line=1, column=10)
    message = Message(text="Error message", code="error_code", start_position=start_position, end_position=end_position)
    assert repr(message) == f"Message(text='Error message', code='error_code', start_position={start_position!r}, end_position={end_position!r})"

def test_message_repr_with_default_code():
    message = Message(text="Error message")
    assert repr(message) == "Message(text='Error message', code='custom')"


# LLM-generated content at query #21
#--------------------------

```python
def test_validation_result_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_validation_result_iter_with_error():
    error = ValidationError("test error")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error


# LLM-generated content at query #22
#--------------------------

```python
def test_iter_with_value():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test_iter_with_error():
    error = ValidationError("Invalid data")
    result = ValidationResult(error=error)
    value, error = result
    assert value is None
    assert error == error

def test_iter_with_none():
    result = ValidationResult()
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #23
#--------------------------

```python
def test_start_position_not_equal_to_end_position():
    message = Message(
        text="Test message",
        code="test_code",
        start_position=Position(line=1, column=1),
        end_position=Position(line=2, column=2)
    )
    assert message.start_position != message.end_position


