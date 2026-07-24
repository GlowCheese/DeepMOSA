####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_same_values():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 3)
    assert pos1 == pos2

def test_eq_different_line_no():
    pos1 = Position(1, 2, 3)
    pos2 = Position(4, 2, 3)
    assert not (pos1 == pos2)

def test_eq_different_column_no():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 5, 3)
    assert not (pos1 == pos2)

def test_eq_different_char_index():
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 6)
    assert not (pos1 == pos2)

def test_eq_all_different():
    pos1 = Position(1, 2, 3)
    pos2 = Position(4, 5, 6)
    assert not (pos1 == pos2)

def test_eq_not_instance_of_position():
    pos = Position(1, 2, 3)
    other = (1, 2, 3)
    assert not (pos == other)

def test_eq_with_none():
    pos = Position(1, 2, 3)
    assert not (pos == None)

def test_eq_same_object():
    pos = Position(1, 2, 3)
    assert pos == pos


# LLM-generated content at query #2
#--------------------------

def test_eq_with_same_values():
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    result = msg1 == msg2
    assert result is True

def test_eq_with_different_text():
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Different", code="custom")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_code():
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="max_length")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_index():
    msg1 = Message(text="Error", index=["users", 0])
    msg2 = Message(text="Error", index=["users", 1])
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_start_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos2, end_position=pos2)
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_end_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=1, column_no=2, char_index=1)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    result = msg1 == msg2
    assert result is False

def test_eq_with_position_vs_start_end():
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    result = msg1 == msg2
    assert result is True

def test_eq_with_none_positions():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True

def test_eq_with_non_message():
    msg = Message(text="Error")
    other = "not a message"
    result = msg == other
    assert result is False

def test_eq_with_key_and_index():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    result = msg1 == msg2
    assert result is True

def test_eq_with_empty_index_and_none():
    msg1 = Message(text="Error", index=[])
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True


# LLM-generated content at query #3
#--------------------------

def test___iter___yields_value_and_error():
    result = ValidationResult(value="test_value")
    iterator = iter(result)
    first = next(iterator)
    second = next(iterator)
    assert first == "test_value"
    assert second is None

def test___iter___yields_none_and_error():
    error = ValidationError()
    result = ValidationResult(error=error)
    iterator = iter(result)
    first = next(iterator)
    second = next(iterator)
    assert first is None
    assert second == error

def test___iter___can_be_unpacked():
    result = ValidationResult(value=123)
    value, error = result
    assert value == 123
    assert error is None

def test___iter___unpacks_error():
    error = ValidationError()
    result = ValidationResult(error=error)
    value, error_unpacked = result
    assert value is None
    assert error_unpacked == error

def test___iter___works_in_for_loop():
    result = ValidationResult(value=[1, 2, 3])
    collected = []
    for item in result:
        collected.append(item)
    assert collected == [[1, 2, 3], None]


# LLM-generated content at query #4
#--------------------------

def test_base_error_constructor_with_single_message():
    error = BaseError(text="Error message", code="custom", key="field")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == ["field"]
    assert error._messages[0].start_position is None
    assert error._messages[0].end_position is None
    assert dict(error) == {"field": "Error message"}

def test_base_error_constructor_with_single_message_and_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    error = BaseError(text="Error message", code="custom", key="field", position=position)
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == ["field"]
    assert error._messages[0].start_position == position
    assert error._messages[0].end_position == position
    assert dict(error) == {"field": "Error message"}

def test_base_error_constructor_with_single_message_no_key():
    error = BaseError(text="Error message", code="custom")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == []
    assert error._messages[0].start_position is None
    assert error._messages[0].end_position is None
    assert dict(error) == {"": "Error message"}

def test_base_error_constructor_with_multiple_messages():
    message1 = Message(text="Error 1", code="code1", key="field1")
    message2 = Message(text="Error 2", code="code2", key="field2")
    error = BaseError(messages=[message1, message2])
    assert len(error._messages) == 2
    assert error._messages[0] == message1
    assert error._messages[1] == message2
    assert dict(error) == {"field1": "Error 1", "field2": "Error 2"}

def test_base_error_constructor_with_nested_index_messages():
    message1 = Message(text="Error 1", code="code1", index=["users", 0, "name"])
    message2 = Message(text="Error 2", code="code2", index=["users", 1, "email"])
    error = BaseError(messages=[message1, message2])
    assert len(error._messages) == 2
    assert error._messages[0] == message1
    assert error._messages[1] == message2
    assert dict(error) == {"users": {0: {"name": "Error 1"}, 1: {"email": "Error 2"}}}

def test_base_error_constructor_with_messages_contradicts_single_args():
    message = Message(text="Error", code="code", key="field")
    try:
        BaseError(text="Another", messages=[message])
    except AssertionError:
        pass

def test_base_error_constructor_with_single_message_contradicts_messages():
    try:
        BaseError(messages=[])
    except AssertionError:
        pass

def test_base_error_constructor_with_no_args():
    try:
        BaseError()
    except AssertionError:
        pass

def test_base_error_constructor_with_only_text():
    error = BaseError(text="Error message")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == []
    assert dict(error) == {"": "Error message"}

def test_base_error_constructor_with_text_and_code():
    error = BaseError(text="Error message", code="max_length")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "max_length"
    assert error._messages[0].index == []
    assert dict(error) == {"": "Error message"}

def test_base_error_constructor_with_text_and_key():
    error = BaseError(text="Error message", key="username")
    assert len(error._messages) == 1
    assert error._messages[0].text == "Error message"
    assert error._messages[0].code == "custom"
    assert error._messages[0].index == ["username"]
    assert dict(error) == {"username": "Error message"}


# LLM-generated content at query #5
#--------------------------

def test_repr_single_message_without_index():
    error = BaseError(text="Error message", code="error_code")
    result = repr(error)
    expected = "BaseError(text='Error message', code='error_code')"
    assert result == expected

def test_repr_single_message_with_index():
    message = Message(text="Error message", code="error_code", index=["key"])
    error = BaseError(messages=[message])
    result = repr(error)
    expected = "BaseError([Message(text='Error message', code='error_code', index=['key'])])"
    assert result == expected

def test_repr_multiple_messages():
    message1 = Message(text="First error", code="code1")
    message2 = Message(text="Second error", code="code2", index=["field"])
    error = BaseError(messages=[message1, message2])
    result = repr(error)
    expected = "BaseError([Message(text='First error', code='code1', index=[]), Message(text='Second error', code='code2', index=['field'])])"
    assert result == expected

def test_repr_empty_messages_raises_assertion():
    try:
        BaseError(messages=[])
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"

def test_repr_mixed_instantiation_raises_assertion():
    try:
        BaseError(text="Error", messages=[Message(text="Error")])
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_with_same_values():
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    result = msg1 == msg2
    assert result is True

def test_eq_with_different_text():
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Different", code="custom")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_code():
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="max_length")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_index():
    msg1 = Message(text="Error", index=["users", 0])
    msg2 = Message(text="Error", index=["users", 1])
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_start_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos2, end_position=pos1)
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_end_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=1, column_no=2, char_index=1)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    result = msg1 == msg2
    assert result is False

def test_eq_with_same_positions_via_position():
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    result = msg1 == msg2
    assert result is True

def test_eq_with_non_message_instance():
    msg = Message(text="Error")
    result = msg == "not a message"
    assert result is False

def test_eq_with_none_index():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    result = msg1 == msg2
    assert result is True

def test_eq_with_empty_index():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    result = msg1 == msg2
    assert result is True

def test_eq_with_none_positions():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", start_position=None, end_position=None)
    result = msg1 == msg2
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test_str_single_message_without_index():
    error = BaseError(text="An error occurred", code="error_code")
    result = str(error)
    expected = "An error occurred"
    assert result == expected

def test_str_single_message_with_index():
    error = BaseError(text="Field error", code="field_error", key="username")
    result = str(error)
    expected = "Field error"
    assert result == expected

def test_str_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", index=["field1"]),
        Message(text="Error 2", code="code2", index=["field2"])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = str({"field1": "Error 1", "field2": "Error 2"})
    assert result == expected

def test_str_nested_messages():
    messages = [
        Message(text="Nested error", code="nested", index=["parent", "child"])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = str({"parent": {"child": "Nested error"}})
    assert result == expected

def test_str_empty_index():
    messages = [Message(text="Root error", code="root", index=[])]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "Root error"
    assert result == expected

def test_str_mixed_index_messages():
    messages = [
        Message(text="Error A", code="A", index=["x"]),
        Message(text="Error B", code="B", index=[])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = str({"x": "Error A", "": "Error B"})
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_repr_single_message_no_index():
    error = BaseError(text="Error message", code="error_code")
    result = repr(error)
    expected = "BaseError(text='Error message', code='error_code')"
    assert result == expected

def test_repr_single_message_with_index():
    message = Message(text="Error message", code="error_code", index=["key1", "key2"])
    error = BaseError(messages=[message])
    result = repr(error)
    expected = "BaseError([Message(text='Error message', code='error_code', index=['key1', 'key2'])])"
    assert result == expected

def test_repr_multiple_messages():
    message1 = Message(text="Error 1", code="code1", index=["key1"])
    message2 = Message(text="Error 2", code="code2", index=["key2"])
    error = BaseError(messages=[message1, message2])
    result = repr(error)
    expected = "BaseError([Message(text='Error 1', code='code1', index=['key1']), Message(text='Error 2', code='code2', index=['key2'])])"
    assert result == expected

def test_repr_no_code():
    error = BaseError(text="Error message")
    result = repr(error)
    expected = "BaseError(text='Error message', code=None)"
    assert result == expected

def test_repr_empty_messages_list():
    error = BaseError(messages=[])
    result = repr(error)
    expected = "BaseError([])"
    assert result == expected


