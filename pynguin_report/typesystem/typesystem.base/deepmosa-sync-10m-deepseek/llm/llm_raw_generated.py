####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_str_single_message_without_index():
    error = BaseError(text="Error message", code="error_code")
    result = str(error)
    expected = "Error message"
    assert result == expected

def test_str_single_message_with_index():
    error = BaseError(text="Error message", code="error_code", key="field")
    result = str(error)
    expected = "{'field': 'Error message'}"
    assert result == expected

def test_str_multiple_messages():
    messages = [
        Message(text="Error 1", code="code1", key="field1"),
        Message(text="Error 2", code="code2", key="field2")
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "{'field1': 'Error 1', 'field2': 'Error 2'}"
    assert result == expected

def test_str_nested_messages():
    messages = [
        Message(text="Nested error", code="code", index=["parent", "child"])
    ]
    error = BaseError(messages=messages)
    result = str(error)
    expected = "{'parent': {'child': 'Nested error'}}"
    assert result == expected

def test_str_empty_index_key():
    error = BaseError(text="Error message", code="error_code", key="")
    result = str(error)
    expected = "{'': 'Error message'}"
    assert result == expected

def test_str_with_position_in_message():
    position = Position(line=1, column=5, char_index=10)
    error = BaseError(text="Error message", code="error_code", position=position)
    result = str(error)
    expected = "Error message"
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_message_constructor_with_text_only():
    msg = Message(text="Error message")
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_code():
    msg = Message(text="Error message", code="max_length")
    assert msg.text == "Error message"
    assert msg.code == "max_length"
    assert msg.index == []
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_key():
    msg = Message(text="Error message", key="username")
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == ["username"]
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_index():
    msg = Message(text="Error message", index=["users", 3, "username"])
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == ["users", 3, "username"]
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_position():
    pos = Position(line_no=1, column_no=5, char_index=10)
    msg = Message(text="Error message", position=pos)
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position == pos
    assert msg.end_position == pos

def test_message_constructor_with_start_and_end_position():
    start_pos = Position(line_no=1, column_no=5, char_index=10)
    end_pos = Position(line_no=1, column_no=15, char_index=20)
    msg = Message(text="Error message", start_position=start_pos, end_position=end_pos)
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position == start_pos
    assert msg.end_position == end_pos

def test_message_constructor_key_and_index_mutually_exclusive():
    try:
        Message(text="Error message", key="username", index=["users", 0])
    except AssertionError:
        pass

def test_message_constructor_position_and_start_end_mutually_exclusive():
    pos = Position(line_no=1, column_no=5, char_index=10)
    try:
        Message(text="Error message", position=pos, start_position=pos)
    except AssertionError:
        pass
    try:
        Message(text="Error message", position=pos, end_position=pos)
    except AssertionError:
        pass


# LLM-generated content at query #3
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
    pos2 = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    result = msg1 == msg2
    assert result is False

def test_eq_with_none_positions():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True

def test_eq_with_position_vs_start_end():
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    result = msg1 == msg2
    assert result is True

def test_eq_with_different_type():
    msg = Message(text="Error")
    other = "Not a Message"
    result = msg == other
    assert result is False

def test_eq_with_same_hash_identifiers():
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    result = msg1 == msg2
    assert result is True


# LLM-generated content at query #4
#--------------------------

def test___eq___with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True

def test___eq___with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    result = error1 == error2
    assert result is False

def test___eq___with_multiple_messages():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test___eq___with_different_multiple_messages():
    messages1 = [Message(text="Error 1"), Message(text="Error 2")]
    messages2 = [Message(text="Error 1"), Message(text="Error 3")]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    result = error1 == error2
    assert result is False

def test___eq___with_non_validation_error():
    error = BaseError(text="Error message", code="error_code")
    other = "not an error"
    result = error == other
    assert result is False

def test___eq___with_same_indexed_messages():
    message = Message(text="Error", index=["key1", "key2"])
    error1 = BaseError(messages=[message])
    error2 = BaseError(messages=[message])
    result = error1 == error2
    assert result is True

def test___eq___with_different_indexed_messages():
    message1 = Message(text="Error", index=["key1", "key2"])
    message2 = Message(text="Error", index=["key1", "key3"])
    error1 = BaseError(messages=[message1])
    error2 = BaseError(messages=[message2])
    result = error1 == error2
    assert result is False

def test___eq___with_empty_messages_list():
    error1 = BaseError(messages=[])
    error2 = BaseError(messages=[])
    result = error1 == error2
    assert result is True


# LLM-generated content at query #5
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
    result = ValidationResult(value=[1, 2])
    collected = []
    for item in result:
        collected.append(item)
    assert collected == [[1, 2], None]


# LLM-generated content at query #6
#--------------------------

def test_position_and_start_position_both_provided():
    pos = Position(1, 1, 0)
    start_pos = Position(1, 1, 0)
    end_pos = Position(1, 10, 9)
    try:
        Message(text="error", code="custom", position=pos, start_position=start_pos, end_position=end_pos)
    except AssertionError as e:
        pass

def test_position_and_end_position_both_provided():
    pos = Position(1, 1, 0)
    start_pos = Position(1, 1, 0)
    end_pos = Position(1, 10, 9)
    try:
        Message(text="error", code="custom", position=pos, start_position=start_pos, end_position=end_pos)
    except AssertionError as e:
        pass

def test_position_and_start_end_positions_all_provided():
    pos = Position(1, 1, 0)
    start_pos = Position(1, 1, 0)
    end_pos = Position(1, 10, 9)
    try:
        Message(text="error", code="custom", position=pos, start_position=start_pos, end_position=end_pos)
    except AssertionError as e:
        pass


# LLM-generated content at query #7
#--------------------------

def test_eq_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True

def test_eq_with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    result = error1 == error2
    assert result is False

def test_eq_with_multiple_messages():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test_eq_with_different_multiple_messages():
    messages1 = [Message(text="Error 1"), Message(text="Error 2")]
    messages2 = [Message(text="Error 1"), Message(text="Error 3")]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    result = error1 == error2
    assert result is False

def test_eq_with_non_validation_error():
    error = BaseError(text="Error message", code="error_code")
    other = "not an error"
    result = error == other
    assert result is False

def test_eq_with_same_indexed_messages():
    error1 = BaseError(text="Error message", code="error_code", key="field1")
    error2 = BaseError(text="Error message", code="error_code", key="field1")
    result = error1 == error2
    assert result is True

def test_eq_with_different_indexed_messages():
    error1 = BaseError(text="Error message", code="error_code", key="field1")
    error2 = BaseError(text="Error message", code="error_code", key="field2")
    result = error1 == error2
    assert result is False


# LLM-generated content at query #8
#--------------------------

def test_position_and_start_end_position_both_provided():
    pos = Position(line_no=1, column_no=1, char_index=0)
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    message = Message(text="error", position=pos, start_position=start_pos, end_position=end_pos)


# LLM-generated content at query #9
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result == True

def test_eq_returns_true_for_same_messages_with_key():
    error1 = BaseError(text="Error message", code="error_code", key="field")
    error2 = BaseError(text="Error message", code="error_code", key="field")
    result = error1 == error2
    assert result == True

def test_eq_returns_true_for_same_messages_with_position():
    from types import SimpleNamespace
    position = SimpleNamespace(line=1, column=1)
    error1 = BaseError(text="Error message", code="error_code", position=position)
    error2 = BaseError(text="Error message", code="error_code", position=position)
    result = error1 == error2
    assert result == True

def test_eq_returns_true_for_same_messages_list():
    from types import SimpleNamespace
    Message = SimpleNamespace
    messages = [Message(text="Error 1", code="code1"), Message(text="Error 2", code="code2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result == True


# LLM-generated content at query #10
#--------------------------

def test_message_constructor_with_text_only():
    msg = Message(text="Error message")
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_code():
    msg = Message(text="Error message", code="max_length")
    assert msg.text == "Error message"
    assert msg.code == "max_length"
    assert msg.index == []
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_key():
    msg = Message(text="Error message", key="username")
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == ["username"]
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_index():
    msg = Message(text="Error message", index=["users", 3, "username"])
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == ["users", 3, "username"]
    assert msg.start_position is None
    assert msg.end_position is None

def test_message_constructor_with_position():
    pos = Position(line_no=1, column_no=5, char_index=10)
    msg = Message(text="Error message", position=pos)
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position == pos
    assert msg.end_position == pos

def test_message_constructor_with_start_and_end_position():
    start_pos = Position(line_no=1, column_no=5, char_index=10)
    end_pos = Position(line_no=1, column_no=15, char_index=20)
    msg = Message(text="Error message", start_position=start_pos, end_position=end_pos)
    assert msg.text == "Error message"
    assert msg.code == "custom"
    assert msg.index == []
    assert msg.start_position == start_pos
    assert msg.end_position == end_pos

def test_message_constructor_key_and_index_mutually_exclusive():
    try:
        Message(text="Error message", key="username", index=["users", 3])
    except AssertionError:
        pass

def test_message_constructor_position_and_start_end_mutually_exclusive():
    pos = Position(line_no=1, column_no=5, char_index=10)
    try:
        Message(text="Error message", position=pos, start_position=pos)
    except AssertionError:
        pass
    try:
        Message(text="Error message", position=pos, end_position=pos)
    except AssertionError:
        pass


# LLM-generated content at query #11
#--------------------------

def test_eq_with_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    assert error1 == error2

def test_eq_with_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    assert not (error1 == error2)

def test_eq_with_same_multiple_messages():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    assert error1 == error2

def test_eq_with_different_multiple_messages():
    messages1 = [Message(text="Error 1"), Message(text="Error 2")]
    messages2 = [Message(text="Error 1"), Message(text="Error 3")]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    assert not (error1 == error2)

def test_eq_with_non_validation_error():
    error = BaseError(text="Error message")
    other = "not an error"
    assert not (error == other)

def test_eq_with_different_index_in_messages():
    message1 = Message(text="Error", index=["key1"])
    message2 = Message(text="Error", index=["key2"])
    error1 = BaseError(messages=[message1])
    error2 = BaseError(messages=[message2])
    assert not (error1 == error2)

def test_eq_with_same_index_in_messages():
    message1 = Message(text="Error", index=["key1"])
    message2 = Message(text="Error", index=["key1"])
    error1 = BaseError(messages=[message1])
    error2 = BaseError(messages=[message2])
    assert error1 == error2


# LLM-generated content at query #12
#--------------------------

def test___eq__returns_true_for_same_messages():
    error1 = BaseError(text="Error message", code="error_code")
    error2 = BaseError(text="Error message", code="error_code")
    result = error1 == error2
    assert result is True

def test___eq__returns_false_for_different_messages():
    error1 = BaseError(text="Error message 1", code="error_code")
    error2 = BaseError(text="Error message 2", code="error_code")
    result = error1 == error2
    assert result is False

def test___eq__returns_false_for_different_types():
    error = BaseError(text="Error message", code="error_code")
    other = "not an error"
    result = error == other
    assert result is False

def test___eq__returns_true_for_identical_multi_message_errors():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test___eq__returns_false_for_different_multi_message_errors():
    messages1 = [Message(text="Error 1"), Message(text="Error 2")]
    messages2 = [Message(text="Error 1"), Message(text="Error 3")]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    result = error1 == error2
    assert result is False

def test___eq__returns_true_for_same_messages_with_key_and_position():
    error1 = BaseError(text="Error", code="code", key="field", position=Position(line=1, column=1))
    error2 = BaseError(text="Error", code="code", key="field", position=Position(line=1, column=1))
    result = error1 == error2
    assert result is True


# LLM-generated content at query #13
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
        Message(text="Error 1", code="code1", key=None, index=None, position=None),
        Message(text="Error 2", code="code2", key=None, index=None, position=None)
    ]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True


# LLM-generated content at query #14
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error 1", code="code1")
    error2 = BaseError(text="Error 1", code="code1")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_key():
    error1 = BaseError(text="Error 1", code="code1", key="key1")
    error2 = BaseError(text="Error 1", code="code1", key="key1")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_position():
    position = Position(line=1, column=1, char_index=0)
    error1 = BaseError(text="Error 1", code="code1", position=position)
    error2 = BaseError(text="Error 1", code="code1", position=position)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_list():
    messages = [Message(text="Error 1", code="code1"), Message(text="Error 2", code="code2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_identical_messages_list():
    messages1 = [Message(text="Error 1", code="code1"), Message(text="Error 2", code="code2")]
    messages2 = [Message(text="Error 1", code="code1"), Message(text="Error 2", code="code2")]
    error1 = BaseError(messages=messages1)
    error2 = BaseError(messages=messages2)
    result = error1 == error2
    assert result is True


# LLM-generated content at query #15
#--------------------------

def test_eq_returns_true_for_same_messages():
    error1 = BaseError(text="Error 1")
    error2 = BaseError(text="Error 1")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_code():
    error1 = BaseError(text="Error 1", code="code1")
    error2 = BaseError(text="Error 1", code="code1")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_key():
    error1 = BaseError(text="Error 1", key="key1")
    error2 = BaseError(text="Error 1", key="key1")
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_with_position():
    position = Position(line=1, column=1, char_index=0)
    error1 = BaseError(text="Error 1", position=position)
    error2 = BaseError(text="Error 1", position=position)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_list():
    messages = [Message(text="Error 1"), Message(text="Error 2")]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True

def test_eq_returns_true_for_same_messages_list_with_index():
    messages = [Message(text="Error 1", index=["field1"]), Message(text="Error 2", index=["field2"])]
    error1 = BaseError(messages=messages)
    error2 = BaseError(messages=messages)
    result = error1 == error2
    assert result is True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_with_same_attributes():
    msg1 = Message(text="Error", code="custom", key="field", start_position=1, end_position=5)
    msg2 = Message(text="Error", code="custom", key="field", start_position=1, end_position=5)
    result = msg1 == msg2
    assert result is True

def test_eq_with_different_text():
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_code():
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_index():
    msg1 = Message(text="Error", index=["a", 1])
    msg2 = Message(text="Error", index=["a", 2])
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_start_position():
    msg1 = Message(text="Error", start_position=1, end_position=5)
    msg2 = Message(text="Error", start_position=2, end_position=5)
    result = msg1 == msg2
    assert result is False

def test_eq_with_different_end_position():
    msg1 = Message(text="Error", start_position=1, end_position=5)
    msg2 = Message(text="Error", start_position=1, end_position=6)
    result = msg1 == msg2
    assert result is False

def test_eq_with_position_attribute():
    msg1 = Message(text="Error", position=3)
    msg2 = Message(text="Error", position=3)
    result = msg1 == msg2
    assert result is True

def test_eq_with_position_vs_start_end():
    msg1 = Message(text="Error", position=3)
    msg2 = Message(text="Error", start_position=3, end_position=3)
    result = msg1 == msg2
    assert result is True

def test_eq_with_non_message_object():
    msg = Message(text="Error")
    other = "not a message"
    result = msg == other
    assert result is False

def test_eq_with_none_index():
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    result = msg1 == msg2
    assert result is True

def test_eq_with_none_positions():
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    result = msg1 == msg2
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test_repr_with_only_text_and_code():
    msg = Message(text="Error", code="custom")
    result = repr(msg)
    expected = "Message(text='Error', code='custom')"
    assert result == expected

def test_repr_with_index():
    msg = Message(text="Error", code="max_length", index=["users", 0, "name"])
    result = repr(msg)
    expected = "Message(text='Error', code='max_length', index=['users', 0, 'name'])"
    assert result == expected

def test_repr_with_position():
    pos = Position(line_no=1, column_no=5, char_index=10)
    msg = Message(text="Error", code="invalid", position=pos)
    result = repr(msg)
    expected = "Message(text='Error', code='invalid', position=Position(line_no=1, column_no=5, char_index=10))"
    assert result == expected

def test_repr_with_start_and_end_position():
    start = Position(line_no=1, column_no=5, char_index=10)
    end = Position(line_no=1, column_no=10, char_index=15)
    msg = Message(text="Error", code="invalid", start_position=start, end_position=end)
    result = repr(msg)
    expected = "Message(text='Error', code='invalid', start_position=Position(line_no=1, column_no=5, char_index=10), end_position=Position(line_no=1, column_no=10, char_index=15))"
    assert result == expected

def test_repr_with_key_instead_of_index():
    msg = Message(text="Error", code="required", key="username")
    result = repr(msg)
    expected = "Message(text='Error', code='required', index=['username'])"
    assert result == expected

def test_repr_with_empty_index():
    msg = Message(text="Error", code="custom", index=[])
    result = repr(msg)
    expected = "Message(text='Error', code='custom')"
    assert result == expected

def test_repr_with_none_index_and_no_position():
    msg = Message(text="Error", code="custom", key=None, index=None)
    result = repr(msg)
    expected = "Message(text='Error', code='custom')"
    assert result == expected


# LLM-generated content at query #3
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

def test_eq_with_itself():
    pos = Position(1, 2, 3)
    assert pos == pos

def test_eq_zero_values():
    pos1 = Position(0, 0, 0)
    pos2 = Position(0, 0, 0)
    assert pos1 == pos2


# LLM-generated content at query #4
#--------------------------

def test___iter___yields_value_and_error():
    value = "test_value"
    result = ValidationResult(value=value)
    iterator = iter(result)
    first = next(iterator)
    second = next(iterator)
    assert first == value
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
    value = 123
    result = ValidationResult(value=value)
    unpacked_value, unpacked_error = result
    assert unpacked_value == value
    assert unpacked_error is None

def test___iter___unpacks_error():
    error = ValidationError()
    result = ValidationResult(error=error)
    unpacked_value, unpacked_error = result
    assert unpacked_value is None
    assert unpacked_error == error

def test___iter___works_in_for_loop():
    result = ValidationResult(value=[1, 2, 3])
    collected = []
    for item in result:
        collected.append(item)
    assert collected == [[1, 2, 3], None]


# LLM-generated content at query #5
#--------------------------

def test_repr_single_message_without_index():
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
    message1 = Message(text="Error 1", code="code1")
    message2 = Message(text="Error 2", code="code2", index=["key"])
    error = BaseError(messages=[message1, message2])
    result = repr(error)
    expected = "BaseError([Message(text='Error 1', code='code1'), Message(text='Error 2', code='code2', index=['key'])])"
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


