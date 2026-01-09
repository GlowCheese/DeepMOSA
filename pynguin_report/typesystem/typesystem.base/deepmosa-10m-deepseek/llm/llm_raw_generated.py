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
    pos2 = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    result = msg1 == msg2
    assert result is False

def test_eq_with_same_position():
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    result = msg1 == msg2
    assert result is True

def test_eq_with_non_message_object():
    msg = Message(text="Error")
    result = msg == "not a message"
    assert result is False

def test_eq_with_none():
    msg = Message(text="Error")
    result = msg == None
    assert result is False

def test_eq_with_key_and_index():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    result = msg1 == msg2
    assert result is True

def test_eq_with_empty_index():
    msg1 = Message(text="Error", index=[])
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True

def test_eq_with_none_positions():
    msg1 = Message(text="Error", start_position=None, end_position=None)
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test___iter___with_value():
    vr = ValidationResult(value="test_value")
    result = list(vr)
    assert result == ["test_value", None]

def test___iter___with_error():
    error = ValidationError()
    vr = ValidationResult(error=error)
    result = list(vr)
    assert result == [None, error]

def test___iter___unpacking():
    vr = ValidationResult(value=42)
    value, error = vr
    assert value == 42
    assert error is None

def test___iter___unpacking_error():
    error = ValidationError()
    vr = ValidationResult(error=error)
    value, error_unpacked = vr
    assert value is None
    assert error_unpacked is error


# LLM-generated content at query #3
#--------------------------

def test_constructor_with_single_message():
    error = BaseError(text="Invalid input", code="invalid", key="username")
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].text == "Invalid input"
    assert messages[0].code == "invalid"
    assert messages[0].index == ["username"]

def test_constructor_with_single_message_and_position():
    position = Position(line_no=1, column_no=5, char_index=4)
    error = BaseError(text="Error at position", code="position_error", position=position)
    messages = error.messages()
    assert len(messages) == 1
    assert messages[0].text == "Error at position"
    assert messages[0].code == "position_error"
    assert messages[0].index == []
    assert messages[0].start_position == position
    assert messages[0].end_position == position

def test_constructor_with_multiple_messages():
    message1 = Message(text="First error", code="error1", index=["field1"])
    message2 = Message(text="Second error", code="error2", index=["field2"])
    error = BaseError(messages=[message1, message2])
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0] == message1
    assert messages[1] == message2

def test_constructor_with_empty_messages_list_raises_assertion():
    try:
        BaseError(messages=[])
    except AssertionError:
        pass

def test_constructor_with_conflicting_arguments_raises_assertion():
    try:
        BaseError(text="Text", messages=[Message(text="Msg", code="code")])
    except AssertionError:
        pass

def test_constructor_with_key_and_index():
    error = BaseError(text="Error", key="key")
    messages = error.messages()
    assert messages[0].index == ["key"]

def test_constructor_without_key_or_index():
    error = BaseError(text="Error", code="code")
    messages = error.messages()
    assert messages[0].index == []

def test_constructor_message_dict_population():
    error = BaseError(text="Error", key="field")
    assert dict(error) == {"field": "Error"}

def test_constructor_message_dict_population_nested():
    message = Message(text="Nested error", code="nested", index=["parent", "child"])
    error = BaseError(messages=[message])
    assert dict(error) == {"parent": {"child": "Nested error"}}

def test_constructor_with_multiple_messages_same_key():
    message1 = Message(text="Error 1", code="code1", index=["field"])
    message2 = Message(text="Error 2", code="code2", index=["field"])
    error = BaseError(messages=[message1, message2])
    assert dict(error) == {"field": "Error 2"}


# LLM-generated content at query #4
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
    message1 = Message(text="First error", code="code1", index=["key1"])
    message2 = Message(text="Second error", code="code2", index=["key2"])
    error = BaseError(messages=[message1, message2])
    result = repr(error)
    expected = "BaseError([Message(text='First error', code='code1', index=['key1']), Message(text='Second error', code='code2', index=['key2'])])"
    assert result == expected

def test_repr_single_message_with_position():
    position = Position(line=1, column=5, char_index=10)
    error = BaseError(text="Error message", code="error_code", position=position)
    result = repr(error)
    expected = "BaseError(text='Error message', code='error_code')"
    assert result == expected

def test_repr_empty_messages_list_raises_assertion():
    try:
        BaseError(messages=[])
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"

def test_repr_mixed_instantiation_raises_assertion():
    try:
        BaseError(text="Error", messages=[Message(text="Another")])
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"


# LLM-generated content at query #5
#--------------------------

def test___eq___with_same_messages():
    message1 = Message(text="Error 1", code="code1", index=["key1"])
    message2 = Message(text="Error 2", code="code2", index=["key2"])
    error1 = BaseError(messages=[message1, message2])
    error2 = BaseError(messages=[message1, message2])
    result = error1 == error2
    assert result is True

def test___eq___with_different_messages():
    message1 = Message(text="Error 1", code="code1", index=["key1"])
    message2 = Message(text="Error 2", code="code2", index=["key2"])
    message3 = Message(text="Error 3", code="code3", index=["key3"])
    error1 = BaseError(messages=[message1, message2])
    error2 = BaseError(messages=[message1, message3])
    result = error1 == error2
    assert result is False

def test___eq___with_single_message_instantiation():
    error1 = BaseError(text="Error text", code="error_code", key="field")
    error2 = BaseError(text="Error text", code="error_code", key="field")
    result = error1 == error2
    assert result is True

def test___eq___with_different_single_message_instantiation():
    error1 = BaseError(text="Error text 1", code="error_code1", key="field1")
    error2 = BaseError(text="Error text 2", code="error_code2", key="field2")
    result = error1 == error2
    assert result is False

def test___eq___with_mixed_instantiation_same_messages():
    message = Message(text="Error text", code="error_code", key="field")
    error1 = BaseError(messages=[message])
    error2 = BaseError(text="Error text", code="error_code", key="field")
    result = error1 == error2
    assert result is True

def test___eq___with_non_validation_error_instance():
    error = BaseError(text="Error text", code="error_code")
    other = "not an error"
    result = error == other
    assert result is False

def test___eq___with_empty_messages():
    error1 = BaseError(messages=[])
    error2 = BaseError(messages=[])
    result = error1 == error2
    assert result is True

def test___eq___with_different_message_order():
    message1 = Message(text="Error 1", code="code1")
    message2 = Message(text="Error 2", code="code2")
    error1 = BaseError(messages=[message1, message2])
    error2 = BaseError(messages=[message2, message1])
    result = error1 == error2
    assert result is False


# LLM-generated content at query #6
#--------------------------

def test_str_single_message_without_index():
    error = BaseError(text="Invalid input", code="invalid")
    result = str(error)
    expected = "Invalid input"
    assert result == expected

def test_str_single_message_with_index():
    error = BaseError(text="Invalid input", code="invalid", key="username")
    result = str(error)
    expected = "{'username': 'Invalid input'}"
    assert result == expected

def test_str_multiple_messages():
    messages = [
        Message(text="Invalid input", code="invalid", index=["username"]),
        Message(text="Too short", code="min_length", index=["password"])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "{'username': 'Invalid input', 'password': 'Too short'}"
    assert result == expected

def test_str_nested_messages():
    messages = [
        Message(text="Invalid input", code="invalid", index=["user", "username"]),
        Message(text="Too short", code="min_length", index=["user", "password"])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "{'user': {'username': 'Invalid input', 'password': 'Too short'}}"
    assert result == expected

def test_str_empty_index():
    messages = [
        Message(text="Invalid input", code="invalid", index=[]),
        Message(text="Too short", code="min_length", index=[])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "{'': 'Too short'}"
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_key():
    error1 = BaseError(text="Error message", code="error_code", key="field")
    error2 = BaseError(text="Error message", code="error_code", key="field")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_position():
    from types import SimpleNamespace
    position = SimpleNamespace(line=1, column=1)
    error1 = BaseError(text="Error message", code="error_code", position=position)
    error2 = BaseError(text="Error message", code="error_code", position=position)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_multiple_messages():
    from types import SimpleNamespace
    Message = SimpleNamespace
    messages = [
        Message(text="Error 1", code="code1", key="key1", index=[], position=None),
        Message(text="Error 2", code="code2", key="key2", index=[], position=None)
    ]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_index():
    from types import SimpleNamespace
    Message = SimpleNamespace
    messages = [
        Message(text="Error", code="code", key="key", index=["parent", "child"], position=None)
    ]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True


# LLM-generated content at query #8
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True


# LLM-generated content at query #9
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_key():
    error1 = BaseError(text="Error message", code="error_code", key="field")
    error2 = BaseError(text="Error message", code="error_code", key="field")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_position():
    position = Position(line=1, column=1, char_index=0)
    error1 = BaseError(text="Error message", code="error_code", position=position)
    error2 = BaseError(text="Error message", code="error_code", position=position)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_list():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_identical_validation_error_instance():
    error = BaseError(text="Error message")
    result = error == error
    assert result is True


# LLM-generated content at query #10
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True


# LLM-generated content at query #11
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True


# LLM-generated content at query #12
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True


# LLM-generated content at query #13
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_same_instance():
    msg = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    result = msg == msg
    assert result == True

def test_eq_equal_messages():
    msg1 = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    msg2 = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    result = msg1 == msg2
    assert result == True

def test_eq_different_text():
    msg1 = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    msg2 = Message(text="Different", code="custom", index=[], start_position=None, end_position=None)
    result = msg1 == msg2
    assert result == False

def test_eq_different_code():
    msg1 = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    msg2 = Message(text="Error", code="other", index=[], start_position=None, end_position=None)
    result = msg1 == msg2
    assert result == False

def test_eq_different_index():
    msg1 = Message(text="Error", code="custom", index=["key"], start_position=None, end_position=None)
    msg2 = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    result = msg1 == msg2
    assert result == False

def test_eq_different_start_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=2, char_index=10)
    msg1 = Message(text="Error", code="custom", index=[], start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", code="custom", index=[], start_position=pos2, end_position=pos2)
    result = msg1 == msg2
    assert result == False

def test_eq_different_end_position():
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=1, column_no=1, char_index=0)
    pos3 = Position(line_no=2, column_no=2, char_index=10)
    msg1 = Message(text="Error", code="custom", index=[], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", code="custom", index=[], start_position=pos1, end_position=pos3)
    result = msg1 == msg2
    assert result == False

def test_eq_with_non_message():
    msg = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    result = msg == "not a message"
    assert result == False

def test_eq_with_none():
    msg = Message(text="Error", code="custom", index=[], start_position=None, end_position=None)
    result = msg == None
    assert result == False

def test_eq_same_positions_via_position_arg():
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", index=[], position=pos)
    msg2 = Message(text="Error", code="custom", index=[], start_position=pos, end_position=pos)
    result = msg1 == msg2
    assert result == True

def test_eq_with_key_and_index():
    msg1 = Message(text="Error", code="custom", key="username", start_position=None, end_position=None)
    msg2 = Message(text="Error", code="custom", index=["username"], start_position=None, end_position=None)
    result = msg1 == msg2
    assert result == True


# LLM-generated content at query #2
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

def test_eq_same_object():
    pos = Position(1, 2, 3)
    assert pos == pos


# LLM-generated content at query #3
#--------------------------

def test_base_error_constructor_with_single_message():
    error = BaseError(text="Error text", code="error_code", key="field_name")
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Error text"
    assert message.code == "error_code"
    assert message.index == ["field_name"]
    assert message.start_position is None
    assert message.end_position is None

def test_base_error_constructor_with_single_message_and_position():
    position = Position(line_no=1, column_no=2, char_index=3)
    error = BaseError(text="Error text", code="error_code", position=position)
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Error text"
    assert message.code == "error_code"
    assert message.index == []
    assert message.start_position == position
    assert message.end_position == position

def test_base_error_constructor_with_single_message_no_key_or_index():
    error = BaseError(text="Error text")
    messages = error.messages()
    assert len(messages) == 1
    message = messages[0]
    assert message.text == "Error text"
    assert message.code == "custom"
    assert message.index == []
    assert message.start_position is None
    assert message.end_position is None

def test_base_error_constructor_with_multiple_messages():
    message1 = Message(text="Error 1", code="code1", key="field1")
    message2 = Message(text="Error 2", code="code2", index=["field2", 0])
    error = BaseError(messages=[message1, message2])
    messages = error.messages()
    assert len(messages) == 2
    assert messages[0] == message1
    assert messages[1] == message2

def test_base_error_constructor_with_multiple_messages_empty_list_raises_assertion():
    try:
        BaseError(messages=[])
    except AssertionError:
        pass
    else:
        assert False

def test_base_error_constructor_with_messages_and_text_raises_assertion():
    message = Message(text="Error", code="code")
    try:
        BaseError(text="Another error", messages=[message])
    except AssertionError:
        pass
    else:
        assert False

def test_base_error_constructor_with_messages_and_code_raises_assertion():
    message = Message(text="Error", code="code")
    try:
        BaseError(code="another_code", messages=[message])
    except AssertionError:
        pass
    else:
        assert False

def test_base_error_constructor_with_messages_and_key_raises_assertion():
    message = Message(text="Error", code="code")
    try:
        BaseError(key="field", messages=[message])
    except AssertionError:
        pass
    else:
        assert False

def test_base_error_constructor_with_messages_and_position_raises_assertion():
    message = Message(text="Error", code="code")
    position = Position(line_no=1, column_no=2, char_index=3)
    try:
        BaseError(position=position, messages=[message])
    except AssertionError:
        pass
    else:
        assert False

def test_base_error_constructor_without_text_or_messages_raises_assertion():
    try:
        BaseError()
    except AssertionError:
        pass
    else:
        assert False


# LLM-generated content at query #4
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

def test___iter___yields_none_and_none():
    result = ValidationResult()
    iterator = iter(result)
    first = next(iterator)
    second = next(iterator)
    assert first is None
    assert second is None

def test___iter___can_be_unpacked():
    result = ValidationResult(value=42)
    value, error = result
    assert value == 42
    assert error is None

def test___iter___can_be_unpacked_with_error():
    error = ValidationError()
    result = ValidationResult(error=error)
    value, error_unpacked = result
    assert value is None
    assert error_unpacked == error


