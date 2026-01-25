####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base_error_single_message():
    error = BaseError(text="Error message", code="error_code", key="error_key", position=Position(1, 2, 3))
    assert error._messages == [Message(text="Error message", code="error_code", key="error_key", position=Position(1, 2, 3))]
    assert error._message_dict == {"error_key": "Error message"}

def test_base_error_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", key="key1"),
        Message(text="Error 2", code="code2", index=["parent", "key2"])
    ]
    error = BaseError(messages=messages)
    assert error._messages == messages
    assert error._message_dict == {"key1": "Error 1", "parent": {"key2": "Error 2"}}


# LLM-generated content at query #2
#--------------------------

```python
def test___eq___returns_true_for_equal_validation_errors():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test___eq___returns_false_for_different_validation_errors():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Different message", code="error_code")
    assert not (error1 == error2)

def test___eq___returns_false_for_non_validation_error():
    error = BaseError(text="Error message", code="error_code")
    assert not (error == "not an error")


# LLM-generated content at query #3
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

def test_message_equality_with_different_start_and_end_positions():
    start_pos1 = Position(line=1, column=1)
    end_pos1 = Position(line=1, column=5)
    start_pos2 = Position(line=2, column=2)
    end_pos2 = Position(line=2, column=6)
    msg1 = Message(text="Error", code="test", start_position=start_pos1, end_position=end_pos1)
    msg2 = Message(text="Error", code="test", start_position=start_pos2, end_position=end_pos2)
    assert not (msg1 == msg2)

def test_message_equality_with_none_vs_default_code():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2

def test_message_equality_with_none_vs_empty_index():
    msg1 = Message(text="Error", code="test")
    msg2 = Message(text="Error", code="test", index=[])
    assert msg1 == msg2

def test_message_equality_with_non_message_object():
    msg = Message(text="Error", code="test", key="field")
    assert not (msg == "not a message")


# LLM-generated content at query #4
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1", code="error", key="field")
    msg2 = Message(text="Error 2", code="error", key="field")
    assert not (msg1 == msg2)


# LLM-generated content at query #5
#--------------------------

```python
def test_message_equality_with_different_text():
    msg1 = Message(text="Error 1", code="error", key="field")
    msg2 = Message(text="Error 2", code="error", key="field")
    assert not (msg1 == msg2)


# LLM-generated content at query #6
#--------------------------

```python
def test_message_equality_with_different_text():
    message1 = Message(text="Error message 1")
    message2 = Message(text="Error message 2")
    assert not (message1 == message2)


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_eq_with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code_1")
    error2 = BaseError(text="Error message 2", code="error_code_2")
    assert not (error1 == error2)

def test_eq_with_different_types():
    error = BaseError(text="Error message", code="error_code")
    assert not (error == "not an error")

def test_eq_with_different_number_of_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(messages=[Message(text="Error message", code="error_code"), Message(text="Another error", code="another_code")])
    assert not (error1 == error2)

def test_eq_with_same_messages_different_order():
    error1 = BaseError(messages=[Message(text="Error message", code="error_code"), Message(text="Another error", code="another_code")])
    error2 = BaseError(messages=[Message(text="Another error", code="another_code"), Message(text="Error message", code="error_code")])
    assert not (error1 == error2)


# LLM-generated content at query #8
#--------------------------

```python
def test_eq_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_eq_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code1")
    error2 = BaseError(text="Error message 2", code="error_code2")
    assert not (error1 == error2)

def test_eq_different_types():
    error = BaseError(text="Error message", code="error_code")
    assert not (error == "not an error")

def test_eq_with_multiple_messages():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    assert error1 == error2

def test_eq_with_different_message_counts():
    error1 = BaseError(messages=[Message(text="Error 1")])
    error2 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    assert not (error1 == error2)


# LLM-generated content at query #9
#--------------------------

```python
def test___eq__with_same_messages():
    error1 = BaseError(messages=[Message(text="Error1"), Message(text="Error2")])
    error2 = BaseError(messages=[Message(text="Error1"), Message(text="Error2")])
    assert error1 == error2

def test___eq__with_different_messages():
    error1 = BaseError(messages=[Message(text="Error1")])
    error2 = BaseError(messages=[Message(text="Error2")])
    assert not (error1 == error2)

def test___eq__with_different_types():
    error = BaseError(messages=[Message(text="Error1")])
    assert not (error == "not an error")


# LLM-generated content at query #10
#--------------------------

```python
def test_message_equality_with_different_text():
    message1 = Message(text="Error 1", code="error")
    message2 = Message(text="Error 2", code="error")
    assert not (message1 == message2)


# LLM-generated content at query #11
#--------------------------

```python
def test_equality_with_same_messages():
    error1 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    error2 = BaseError(messages=[Message(text="Error 1"), Message(text="Error 2")])
    assert error1 == error2


# LLM-generated content at query #12
#--------------------------

```python
def test_message_equality_predicate_false():
    message1 = Message(text="Error", code="custom", key="field")
    message2 = Message(text="Different Error", code="custom", key="field")
    assert not (message1 == message2)


# LLM-generated content at query #13
#--------------------------

```python
def test_equality_with_same_messages():
    error1 = BaseError(messages=[Message(text="Error 1", code="code1")])
    error2 = BaseError(messages=[Message(text="Error 1", code="code1")])
    assert error1 == error2


# LLM-generated content at query #14
#--------------------------

```python
def test_eq_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_eq_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    assert not (error1 == error2)

def test_eq_different_types():
    error = BaseError(text="Error message", code="error_code")
    assert not (error == "not an error")

def test_eq_with_multiple_messages():
    messages = [
        Message(text="Error 1", code="error_code", index=[0]),
        Message(text="Error 2", code="error_code", index=[1])
    ]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    assert error1 == error2

def test_eq_with_different_multiple_messages():
    messages1 = [Message(text="Error 1", code="error_code", index=[0])]
    messages2 = [Message(text="Error 2", code="error_code", index=[1])]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    assert not (error1 == error2)


