####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_message_eq_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    assert msg1 == msg2

def test_message_eq_with_different_text():
    msg1 = Message(text="Error1", code="test", key="field")
    msg2 = Message(text="Error2", code="test", key="field")
    assert not (msg1 == msg2)

def test_message_eq_with_different_code():
    msg1 = Message(text="Error", code="test1", key="field")
    msg2 = Message(text="Error", code="test2", key="field")
    assert not (msg1 == msg2)

def test_message_eq_with_different_index():
    msg1 = Message(text="Error", code="test", key="field1")
    msg2 = Message(text="Error", code="test", key="field2")
    assert not (msg1 == msg2)

def test_message_eq_with_different_start_position():
    msg1 = Message(text="Error", code="test", start_position=Position(0, 0), end_position=Position(0, 5))
    msg2 = Message(text="Error", code="test", start_position=Position(0, 1), end_position=Position(0, 5))
    assert not (msg1 == msg2)

def test_message_eq_with_different_end_position():
    msg1 = Message(text="Error", code="test", start_position=Position(0, 0), end_position=Position(0, 5))
    msg2 = Message(text="Error", code="test", start_position=Position(0, 0), end_position=Position(0, 6))
    assert not (msg1 == msg2)

def test_message_eq_with_position_vs_start_end():
    msg1 = Message(text="Error", code="test", position=Position(0, 0))
    msg2 = Message(text="Error", code="test", start_position=Position(0, 0), end_position=Position(0, 0))
    assert msg1 == msg2

def test_message_eq_with_non_message_object():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == "not a message")

def test_message_eq_with_none():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == None)


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

def test_message_constructor_with_code():
    message = Message(text="Error message", code="max_length")
    assert message.text == "Error message"
    assert message.code == "max_length"
    assert message.index == []
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_key():
    message = Message(text="Error message", key="username")
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == ["username"]
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_index():
    message = Message(text="Error message", index=["users", 3, "username"])
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == ["users", 3, "username"]
    assert message.start_position is None
    assert message.end_position is None

def test_message_constructor_with_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    message = Message(text="Error message", position=position)
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position == position
    assert message.end_position == position

def test_message_constructor_with_start_and_end_position():
    start_position = Position(line_no=1, column_no=2, char_index=3)
    end_position = Position(line_no=4, column_no=5, char_index=6)
    message = Message(text="Error message", start_position=start_position, end_position=end_position)
    assert message.text == "Error message"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position == start_position
    assert message.end_position == end_position


# LLM-generated content at query #3
#--------------------------

```python
def test_position_equality():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 3)
    pos3 = Position(4, 5, 6)
    assert pos1 == pos2
    assert not (pos1 == pos3)
    assert not (pos1 == "not a Position")


# LLM-generated content at query #4
#--------------------------

```python
def test_base_error_constructor_single_message():
    error = BaseError(text="Error message", code="error_code", key="error_key")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "error_code"
    assert error._messages[0].index == ["error_key"]
    assert dict(error) == {"error_key": "Error message"}

def test_base_error_constructor_single_message_with_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    error = BaseError(text="Error message", code="error_code", key="error_key", position=position)
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "error_code"
    assert error._messages[0].index == ["error_key"]
    assert error._messages[0].start_position == position
    assert error._messages[0].end_position == position
    assert dict(error) == {"error_key": "Error message"}

def test_base_error_constructor_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", key="key1"),
        Message(text="Error 2", code="code2", key="key2")
    ]
    error = BaseError(messages=messages)
    assert len(error._messages) == 2
    assert error._messages[0].text == "Error 1"
    assert error._messages[0].code == "code1"
    assert error._messages[0].index == ["key1"]
    assert error._messages[1].text == "Error 2"
    assert error._messages[1].code == "code2"
    assert error._messages[1].index == ["key2"]
    assert dict(error) == {"key1": "Error 1", "key2": "Error 2"}

def test_base_error_constructor_multiple_messages_with_index():
    messages = [
        Message(text="Error 1", code="code1", index=["users", 0, "username"]),
        Message(text="Error 2", code="code2", index=["users", 1, "email"])
    ]
    error = BaseError(messages=messages)
    assert len(error._messages) == 2
    assert error._messages[0].text == "Error 1"
    assert error._messages[0].code == "code1"
    assert error._messages[0].index == ["users", 0, "username"]
    assert error._messages[1].text == "Error 2"
    assert error._messages[1].code == "code2"
    assert error._messages[1].index == ["users", 1, "email"]
    assert dict(error) == {"users": {0: {"username": "Error 1"}, 1: {"email": "Error 2"}}}


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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

def test_iter_with_none():
    result = ValidationResult()
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_ValidationResult___iter___with_value():
    vr = ValidationResult(value=42)
    value, error = vr
    assert value == 42
    assert error is None

def test_ValidationResult___iter___with_error():
    vr = ValidationResult(error="Invalid data")
    value, error = vr
    assert value is None
    assert error == "Invalid data"


# LLM-generated content at query #11
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


# LLM-generated content at query #12
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test___str___single_message_no_index():
    error = BaseError(text="Error message", code="error_code")
    assert str(error) == "Error message"

def test___str___multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", index=[0]),
        Message(text="Error 2", code="code2", index=[1])
    ]
    error = BaseError(messages=messages)
    assert str(error) == "{0: 'Error 1', 1: 'Error 2'}"


# LLM-generated content at query #2
#--------------------------

```python
def test_eq_same_messages():
    error1 = BaseError(messages=[Message(text="Error1", code="code1")])
    error2 = BaseError(messages=[Message(text="Error1", code="code1")])
    assert error1 == error2

def test_eq_different_messages():
    error1 = BaseError(messages=[Message(text="Error1", code="code1")])
    error2 = BaseError(messages=[Message(text="Error2", code="code2")])
    assert not (error1 == error2)

def test_eq_different_types():
    error = BaseError(messages=[Message(text="Error1", code="code1")])
    assert not (error == "not an error")

def test_eq_empty_messages():
    error1 = BaseError(messages=[])
    error2 = BaseError(messages=[])
    assert error1 == error2


# LLM-generated content at query #3
#--------------------------

```python
def test_repr_single_message_no_index():
    error = BaseError(text="Error message", code="error_code")
    assert repr(error) == "BaseError(text='Error message', code='error_code')"

def test_repr_single_message_with_index():
    error = BaseError(text="Error message", code="error_code", key="field")
    assert repr(error) == "BaseError([Message(text='Error message', code='error_code', index=['field'])]))"

def test_repr_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", index=["field1"]),
        Message(text="Error 2", code="code2", index=["field2"])
    ]
    error = BaseError(messages=messages)
    assert repr(error) == f"BaseError({messages!r})"


# LLM-generated content at query #4
#--------------------------

```python
def test_equality_with_same_messages():
    error1 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    error2 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    assert error1 == error2


# LLM-generated content at query #5
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

def test_validation_result_iter_with_none():
    result = ValidationResult()
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #6
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field")
    msg2 = Message(text="Error", code="test", key="field")
    assert msg1 == msg2

def test_message_inequality_with_different_text():
    msg1 = Message(text="Error1", code="test")
    msg2 = Message(text="Error2", code="test")
    assert not (msg1 == msg2)

def test_message_inequality_with_different_code():
    msg1 = Message(text="Error", code="test1")
    msg2 = Message(text="Error", code="test2")
    assert not (msg1 == msg2)

def test_message_inequality_with_different_index():
    msg1 = Message(text="Error", code="test", key="field1")
    msg2 = Message(text="Error", code="test", key="field2")
    assert not (msg1 == msg2)

def test_message_inequality_with_different_start_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="test", start_position=pos1)
    msg2 = Message(text="Error", code="test", start_position=pos2)
    assert not (msg1 == msg2)

def test_message_inequality_with_different_end_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="test", end_position=pos1)
    msg2 = Message(text="Error", code="test", end_position=pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_none_vs_default_code():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

def test_message_equality_with_position_vs_start_end():
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    assert msg1 == msg2

def test_message_inequality_with_non_message_object():
    msg = Message(text="Error")
    assert not (msg == "not a message")


# LLM-generated content at query #7
#--------------------------

```python
def test_message_equality_with_different_text():
    message1 = Message(text="Error message 1")
    message2 = Message(text="Error message 2")
    assert not message1 == message2


# LLM-generated content at query #8
#--------------------------

```python
def test_message_equality_with_different_text():
    message1 = Message(text="Error 1")
    message2 = Message(text="Error 2")
    assert not (message1 == message2)


# LLM-generated content at query #9
#--------------------------

```python
def test_eq_returns_true_for_equal_validation_errors():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2


# LLM-generated content at query #10
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1", code="error", key="field")
    msg2 = Message(text="Error 2", code="error", key="field")
    assert not (msg1 == msg2)


# LLM-generated content at query #11
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

def test_iter_with_none():
    result = ValidationResult()
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #12
#--------------------------

```python
def test__iter__returns_iterator_of_value_and_error():
    value = "test_value"
    error = ValidationError("test_error")
    result_with_value = ValidationResult(value=value)
    result_with_error = ValidationResult(error=error)

    iter_value = iter(result_with_value)
    assert next(iter_value) == value
    assert next(iter_value) is None

    iter_error = iter(result_with_error)
    assert next(iter_error) is None
    assert next(iter_error) == error


# LLM-generated content at query #13
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1", code="error")
    msg2 = Message(text="Error 2", code="error")
    assert not (msg1 == msg2)


# LLM-generated content at query #14
#--------------------------

```python
def test_base_error_equality_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_base_error_equality_with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    assert error1 != error2

def test_base_error_equality_with_different_types():
    error = BaseError(text="Error message", code="error_code")
    assert error != "not an error"

def test_base_error_equality_with_multiple_messages():
    messages = [Message(text="Error 1", code="error_code"), Message(text="Error 2", code="error_code")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    assert error1 == error2

def test_base_error_equality_with_different_message_counts():
    error1 = BaseError(text="Error message", code="error_code")
    messages = [Message(text="Error 1", code="error_code"), Message(text="Error 2", code="error_code")]
    error2 = BaseError(messages=messages)
    assert error1 != error2


# LLM-generated content at query #15
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


# LLM-generated content at query #16
#--------------------------

```python
def test_base_error_equality_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_base_error_equality_with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    assert not (error1 == error2)

def test_base_error_equality_with_different_types():
    error = BaseError(text="Error message", code="error_code")
    other = "not an error"
    assert not (error == other)

def test_base_error_equality_with_multiple_messages():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    assert error1 == error2

def test_base_error_equality_with_different_message_counts():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    assert not (error1 == error2)


# LLM-generated content at query #17
#--------------------------

```python
def test_eq_returns_true_for_equal_validation_errors():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2


# LLM-generated content at query #18
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="test", key="field", position=Position(line=1, column=1))
    msg2 = Message(text="Error", code="test", key="field", position=Position(line=1, column=1))
    assert msg1 == msg2

def test_message_equality_with_different_text():
    msg1 = Message(text="Error1", code="test", key="field", position=Position(line=1, column=1))
    msg2 = Message(text="Error2", code="test", key="field", position=Position(line=1, column=1))
    assert not (msg1 == msg2)

def test_message_equality_with_different_code():
    msg1 = Message(text="Error", code="test1", key="field", position=Position(line=1, column=1))
    msg2 = Message(text="Error", code="test2", key="field", position=Position(line=1, column=1))
    assert not (msg1 == msg2)

def test_message_equality_with_different_index():
    msg1 = Message(text="Error", code="test", key="field1", position=Position(line=1, column=1))
    msg2 = Message(text="Error", code="test", key="field2", position=Position(line=1, column=1))
    assert not (msg1 == msg2)

def test_message_equality_with_different_start_position():
    msg1 = Message(text="Error", code="test", key="field", start_position=Position(line=1, column=1), end_position=Position(line=1, column=5))
    msg2 = Message(text="Error", code="test", key="field", start_position=Position(line=2, column=1), end_position=Position(line=2, column=5))
    assert not (msg1 == msg2)

def test_message_equality_with_different_end_position():
    msg1 = Message(text="Error", code="test", key="field", start_position=Position(line=1, column=1), end_position=Position(line=1, column=5))
    msg2 = Message(text="Error", code="test", key="field", start_position=Position(line=1, column=1), end_position=Position(line=1, column=10))
    assert not (msg1 == msg2)

def test_message_equality_with_non_message_object():
    msg = Message(text="Error", code="test", key="field", position=Position(line=1, column=1))
    assert not (msg == "not a message")

def test_message_equality_with_default_code():
    msg1 = Message(text="Error", key="field", position=Position(line=1, column=1))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(line=1, column=1))
    assert msg1 == msg2


# LLM-generated content at query #19
#--------------------------

```python
def test_message_equality_with_same_attributes():
    msg1 = Message(text="Error", code="custom", key="username")
    msg2 = Message(text="Error", code="custom", key="username")
    assert msg1 == msg2

def test_message_equality_with_different_text():
    msg1 = Message(text="Error1", code="custom")
    msg2 = Message(text="Error2", code="custom")
    assert not (msg1 == msg2)

def test_message_equality_with_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)

def test_message_equality_with_different_index():
    msg1 = Message(text="Error", code="custom", index=["users", 1])
    msg2 = Message(text="Error", code="custom", index=["users", 2])
    assert not (msg1 == msg2)

def test_message_equality_with_different_position():
    pos1 = Position(line=1, column=1)
    pos2 = Position(line=2, column=2)
    msg1 = Message(text="Error", code="custom", position=pos1)
    msg2 = Message(text="Error", code="custom", position=pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_different_start_end_positions():
    start_pos1 = Position(line=1, column=1)
    end_pos1 = Position(line=1, column=5)
    start_pos2 = Position(line=2, column=2)
    end_pos2 = Position(line=2, column=6)
    msg1 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos1)
    msg2 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_none_vs_object():
    msg = Message(text="Error", code="custom")
    assert not (msg == None)

def test_message_equality_with_different_type():
    msg = Message(text="Error", code="custom")
    assert not (msg == "not a message")

def test_message_equality_with_same_position_and_start_end():
    pos = Position(line=1, column=1)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", start_position=pos, end_position=pos)
    assert msg1 == msg2


# LLM-generated content at query #20
#--------------------------

```python
def test_eq_returns_true_for_same_messages():
    error1 = BaseError(messages=[Message(text="Error 1")])
    error2 = BaseError(messages=[Message(text="Error 1")])
    assert error1 == error2


# LLM-generated content at query #21
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)


# LLM-generated content at query #22
#--------------------------

```python
def test_equality_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2


