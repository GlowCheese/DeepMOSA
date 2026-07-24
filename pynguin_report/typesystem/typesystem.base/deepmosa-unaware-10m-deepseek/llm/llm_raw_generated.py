####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert not (msg1 == msg2)

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field"])
    msg2 = Message(text="Error", index=["other"])
    assert not (msg1 == msg2)

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos2, end_position=pos1)
    assert not (msg1 == msg2)

    # Test inequality with different end_position
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert not (msg1 == msg2)

    # Test equality with same positions using position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos1)
    assert msg1 == msg2

    # Test equality with same start/end positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error")
    assert not (msg == "not a message")

    # Test equality with None index vs empty list
    msg1 = Message(text="Error", index=None)
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with key vs index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with None code vs default
    msg1 = Message(text="Error", code=None)
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


# LLM-generated content at query #2
#--------------------------

```python
def test_ValidationResult___repr__():
    # Test with value
    result = ValidationResult(value="test_value")
    assert repr(result) == "ValidationResult(value='test_value')"
    
    # Test with error
    error = ValidationError(text="Error message", code="custom")
    result = ValidationResult(error=error)
    assert repr(result) == "ValidationResult(error=ValidationError(text='Error message', code='custom'))"
    
    # Test with error containing multiple messages
    messages = [
        Message(text="First error", code="max_length"),
        Message(text="Second error", code="min_length")
    ]
    error = ValidationError(messages=messages)
    result = ValidationResult(error=error)
    assert repr(result).startswith("ValidationResult(error=ValidationError([")
    assert "Message(text='First error', code='max_length')" in repr(result)
    assert "Message(text='Second error', code='min_length')" in repr(result)
    
    # Test with error containing indexed message
    error = ValidationError(text="Field error", key="username")
    result = ValidationResult(error=error)
    assert repr(result) == "ValidationResult(error=ValidationError([Message(text='Field error', code='custom', index=['username'])]))"
    
    # Test with None value
    result = ValidationResult(value=None)
    assert repr(result) == "ValidationResult(value=None)"


# LLM-generated content at query #3
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same single message
    error1 = ValidationError(text="Error message", code="custom")
    error2 = ValidationError(text="Error message", code="custom")
    assert error1 == error2
    assert hash(error1) == hash(error2)

    # Test equality with same multiple messages
    messages = [
        Message(text="Error 1", code="code1", key="field1"),
        Message(text="Error 2", code="code2", key="field2"),
    ]
    error3 = ValidationError(messages=messages)
    error4 = ValidationError(messages=messages)
    assert error3 == error4
    assert hash(error3) == hash(error4)

    # Test inequality with different messages
    error5 = ValidationError(text="Error message", code="custom")
    error6 = ValidationError(text="Different message", code="custom")
    assert error5 != error6
    assert hash(error5) != hash(error6)

    # Test inequality with different number of messages
    messages1 = [Message(text="Error 1", code="code1")]
    messages2 = [
        Message(text="Error 1", code="code1"),
        Message(text="Error 2", code="code2"),
    ]
    error7 = ValidationError(messages=messages1)
    error8 = ValidationError(messages=messages2)
    assert error7 != error8
    assert hash(error7) != hash(error8)

    # Test inequality with different types
    error9 = ValidationError(text="Error message")
    assert error9 != "not an error"
    assert error9 != 123
    assert error9 != None

    # Test equality with same messages but different order (should be equal)
    messages3 = [
        Message(text="Error 1", code="code1", key="field1"),
        Message(text="Error 2", code="code2", key="field2"),
    ]
    messages4 = [
        Message(text="Error 2", code="code2", key="field2"),
        Message(text="Error 1", code="code1", key="field1"),
    ]
    error10 = ValidationError(messages=messages3)
    error11 = ValidationError(messages=messages4)
    assert error10 == error11
    assert hash(error10) == hash(error11)

    # Test equality with messages having positions
    pos = Position(line_no=1, column_no=1, char_index=0)
    error12 = ValidationError(text="Error", code="code", position=pos)
    error13 = ValidationError(text="Error", code="code", position=pos)
    assert error12 == error13
    assert hash(error12) == hash(error13)

    # Test inequality with different index paths
    error14 = ValidationError(text="Error", code="code", key="field1")
    error15 = ValidationError(text="Error", code="code", key="field2")
    assert error14 != error15
    assert hash(error14) != hash(error15)

    # Test equality between BaseError subclasses with same messages
    parse_error = ParseError(text="Parse error", code="parse")
    validation_error = ValidationError(text="Parse error", code="parse")
    # These should not be equal since they are different classes
    assert parse_error != validation_error

    # Test self-equality
    error16 = ValidationError(text="Error message")
    assert error16 == error16


# LLM-generated content at query #4
#--------------------------

```python
def test_ValidationResult___repr__():
    # Test with value
    result_with_value = ValidationResult(value="test_value")
    assert repr(result_with_value) == "ValidationResult(value='test_value')"
    
    # Test with error
    error = ValidationError(text="Invalid input", code="invalid")
    result_with_error = ValidationResult(error=error)
    assert repr(result_with_error) == "ValidationResult(error=ValidationError(text='Invalid input', code='invalid'))"
    
    # Test with nested error messages
    messages = [
        Message(text="Field required", code="required", key="username"),
        Message(text="Too short", code="min_length", key="password")
    ]
    nested_error = ValidationError(messages=messages)
    result_with_nested_error = ValidationResult(error=nested_error)
    expected_start = "ValidationResult(error=ValidationError(["
    assert repr(result_with_nested_error).startswith(expected_start)
    
    # Test with empty value
    result_empty_value = ValidationResult(value="")
    assert repr(result_empty_value) == "ValidationResult(value='')"
    
    # Test with numeric value
    result_numeric = ValidationResult(value=42)
    assert repr(result_numeric) == "ValidationResult(value=42)"
    
    # Test with complex error structure
    complex_error = ValidationError(
        messages=[
            Message(
                text="Invalid email",
                code="invalid_email",
                index=["users", 0, "email"],
                position=Position(line_no=1, column_no=5, char_index=4)
            )
        ]
    )
    result_complex = ValidationResult(error=complex_error)
    assert "ValidationResult(error=ValidationError([" in repr(result_complex)
    assert "Invalid email" in repr(result_complex)


# LLM-generated content at query #5
#--------------------------

```python
def test_ValidationResult___repr__():
    # Test with value
    result = ValidationResult(value="test_value")
    assert repr(result) == "ValidationResult(value='test_value')"
    
    # Test with error containing single message without index
    error = ValidationError(text="Invalid input", code="invalid")
    result = ValidationResult(error=error)
    assert repr(result) == "ValidationResult(error=ValidationError(text='Invalid input', code='invalid'))"
    
    # Test with error containing single message with index
    error = ValidationError(text="Field required", code="required", key="username")
    result = ValidationResult(error=error)
    assert repr(result) == "ValidationResult(error=ValidationError([Message(text='Field required', code='required', index=['username'])]))"
    
    # Test with error containing multiple messages
    messages = [
        Message(text="Too short", code="min_length", key="password"),
        Message(text="Invalid format", code="format", key="email")
    ]
    error = ValidationError(messages=messages)
    result = ValidationResult(error=error)
    assert repr(result).startswith("ValidationResult(error=ValidationError([")
    assert "Message(text='Too short'" in repr(result)
    assert "Message(text='Invalid format'" in repr(result)
    
    # Test with None value
    result = ValidationResult(value=None)
    assert repr(result) == "ValidationResult(value=None)"
    
    # Test with complex value
    result = ValidationResult(value={"key": "value", "number": 42})
    assert repr(result) == "ValidationResult(value={'key': 'value', 'number': 42})"


# LLM-generated content at query #6
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert msg1 != msg2

    # Test equality with same position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", position=pos)
    assert msg1 == msg2

    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos2)
    assert msg1 != msg2

    # Test equality with same start/end positions
    start = Position(line_no=1, column_no=1, char_index=0)
    end = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", start_position=start, end_position=end)
    msg2 = Message(text="Error", start_position=start, end_position=end)
    assert msg1 == msg2

    # Test inequality with different start/end positions
    start1 = Position(line_no=1, column_no=1, char_index=0)
    end1 = Position(line_no=1, column_no=5, char_index=4)
    start2 = Position(line_no=2, column_no=1, char_index=10)
    end2 = Position(line_no=2, column_no=5, char_index=14)
    msg1 = Message(text="Error", start_position=start1, end_position=end1)
    msg2 = Message(text="Error", start_position=start2, end_position=end2)
    assert msg1 != msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2

    # Test inequality when one has position and other doesn't
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error")
    assert msg1 != msg2

    # Test equality with complex index
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 0, "name"])
    assert msg1 == msg2

    # Test inequality with complex index difference
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 1, "name"])
    assert msg1 != msg2


# LLM-generated content at query #7
#--------------------------

```python
def test_ValidationResult___iter__():
    # Test with value only
    result = ValidationResult(value="test_value")
    value, error = result
    assert value == "test_value"
    assert error is None
    
    # Test with error only
    validation_error = ValidationError(text="Invalid input")
    result = ValidationResult(error=validation_error)
    value, error = result
    assert value is None
    assert error == validation_error
    
    # Test iteration in for loop
    result = ValidationResult(value=42)
    items = []
    for item in result:
        items.append(item)
    assert items == [42, None]
    
    # Test unpacking with tuple assignment
    result = ValidationResult(value={"key": "value"})
    val, err = result
    assert val == {"key": "value"}
    assert err is None
    
    # Test with None value and error
    result = ValidationResult(value=None)
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #8
#--------------------------

```python
def test_ValidationResult___iter__():
    # Test with value only
    result = ValidationResult(value="test_value")
    value, error = iter(result)
    assert value == "test_value"
    assert error is None
    
    # Test with error only
    error = ValidationError(text="Error message")
    result = ValidationResult(error=error)
    value, error_out = iter(result)
    assert value is None
    assert error_out == error
    
    # Test unpacking directly
    result = ValidationResult(value=42)
    val, err = result
    assert val == 42
    assert err is None
    
    # Test with multiple messages error
    messages = [
        Message(text="Error 1", key="field1"),
        Message(text="Error 2", key="field2")
    ]
    error = ValidationError(messages=messages)
    result = ValidationResult(error=error)
    value, error_out = result
    assert value is None
    assert error_out == error
    
    # Test with complex value
    complex_value = {"key": "value", "list": [1, 2, 3]}
    result = ValidationResult(value=complex_value)
    val, err = iter(result)
    assert val == complex_value
    assert err is None


# LLM-generated content at query #9
#--------------------------

```python
def test_Position___eq__():
    # Test equality with same values
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 3)
    assert pos1 == pos2
    assert not (pos1 != pos2)
    
    # Test inequality with different line_no
    pos3 = Position(2, 2, 3)
    assert pos1 != pos3
    assert not (pos1 == pos3)
    
    # Test inequality with different column_no
    pos4 = Position(1, 3, 3)
    assert pos1 != pos4
    assert not (pos1 == pos4)
    
    # Test inequality with different char_index
    pos5 = Position(1, 2, 4)
    assert pos1 != pos5
    assert not (pos1 == pos5)
    
    # Test equality with self
    assert pos1 == pos1
    
    # Test equality with different object type
    assert pos1 != "not a Position"
    assert not (pos1 == "not a Position")
    
    # Test equality with None
    assert pos1 != None
    assert not (pos1 == None)
    
    # Test equality with subclass (should work since isinstance check)
    class SubPosition(Position):
        pass
    
    subpos = SubPosition(1, 2, 3)
    assert pos1 == subpos
    assert subpos == pos1


# LLM-generated content at query #10
#--------------------------

```python
def test_Position___eq__():
    # Test equality with same values
    pos1 = Position(1, 2, 3)
    pos2 = Position(1, 2, 3)
    assert pos1 == pos2
    assert pos2 == pos1
    
    # Test inequality with different line_no
    pos3 = Position(2, 2, 3)
    assert pos1 != pos3
    assert not (pos1 == pos3)
    
    # Test inequality with different column_no
    pos4 = Position(1, 3, 3)
    assert pos1 != pos4
    assert not (pos1 == pos4)
    
    # Test inequality with different char_index
    pos5 = Position(1, 2, 4)
    assert pos1 != pos5
    assert not (pos1 == pos5)
    
    # Test equality with itself
    assert pos1 == pos1
    
    # Test equality with different Position instances but same values
    pos6 = Position(10, 20, 30)
    pos7 = Position(10, 20, 30)
    assert pos6 == pos7
    
    # Test inequality with completely different values
    pos8 = Position(100, 200, 300)
    assert pos1 != pos8
    
    # Test comparison with non-Position object
    assert pos1 != "not a Position"
    assert pos1 != 123
    assert pos1 != None
    assert pos1 != {"line_no": 1, "column_no": 2, "char_index": 3}
    
    # Test comparison with object that has same attributes but different type
    class FakePosition:
        def __init__(self, line_no, column_no, char_index):
            self.line_no = line_no
            self.column_no = column_no
            self.char_index = char_index
    
    fake_pos = FakePosition(1, 2, 3)
    assert pos1 != fake_pos
    
    # Test with negative values
    pos9 = Position(-1, -2, -3)
    pos10 = Position(-1, -2, -3)
    assert pos9 == pos10
    
    # Test with zero values
    pos11 = Position(0, 0, 0)
    pos12 = Position(0, 0, 0)
    assert pos11 == pos12


# LLM-generated content at query #11
#--------------------------

```python
def test_BaseError___str__():
    # Test single message without index
    error1 = ValidationError(text="Invalid value", code="invalid")
    assert str(error1) == "Invalid value"
    
    # Test single message with index
    error2 = ValidationError(text="Field required", code="required", key="username")
    assert str(error2) == "{'username': 'Field required'}"
    
    # Test multiple messages with simple indices
    error3 = ValidationError(messages=[
        Message(text="Invalid email", code="invalid", key="email"),
        Message(text="Too short", code="min_length", key="password")
    ])
    assert str(error3) == "{'email': 'Invalid email', 'password': 'Too short'}"
    
    # Test nested messages
    error4 = ValidationError(messages=[
        Message(text="Invalid value", code="invalid", index=["users", 0, "email"]),
        Message(text="Required field", code="required", index=["users", 1, "name"])
    ])
    expected4 = "{'users': {0: {'email': 'Invalid value'}, 1: {'name': 'Required field'}}}"
    assert str(error4) == expected4
    
    # Test mixed depth messages
    error5 = ValidationError(messages=[
        Message(text="Invalid format", code="format", key="email"),
        Message(text="Too long", code="max_length", index=["profile", "bio"])
    ])
    expected5 = "{'email': 'Invalid format', 'profile': {'bio': 'Too long'}}"
    assert str(error5) == expected5
    
    # Test empty index (root level message)
    error6 = ValidationError(messages=[
        Message(text="Root error", code="root_error", index=[]),
        Message(text="Field error", code="field_error", key="field")
    ])
    assert str(error6) == "{'': 'Root error', 'field': 'Field error'}"
    
    # Test ParseError (subclass of BaseError)
    error7 = ParseError(text="Parse failed", code="parse_error")
    assert str(error7) == "Parse failed"
    
    # Test with position information (should not affect __str__)
    pos = Position(line_no=1, column_no=5, char_index=4)
    error8 = ValidationError(text="Error with position", code="error", position=pos)
    assert str(error8) == "Error with position"


# LLM-generated content at query #12
#--------------------------

```python
def test_ValidationResult___iter__():
    # Test with value only
    result = ValidationResult(value="test_value")
    value, error = iter(result)
    assert value == "test_value"
    assert error is None
    
    # Test with error only
    validation_error = ValidationError(text="Error message", code="custom")
    result = ValidationResult(error=validation_error)
    value, error = iter(result)
    assert value is None
    assert error == validation_error
    
    # Test unpacking directly
    result = ValidationResult(value=42)
    unpacked_value, unpacked_error = result
    assert unpacked_value == 42
    assert unpacked_error is None
    
    # Test with complex error
    messages = [
        Message(text="Error 1", code="error1", key="field1"),
        Message(text="Error 2", code="error2", key="field2")
    ]
    validation_error = ValidationError(messages=messages)
    result = ValidationResult(error=validation_error)
    value, error = iter(result)
    assert value is None
    assert error == validation_error
    
    # Test with None value and error (edge case - should not happen per constructor)
    # This tests that the iterator yields exactly two items
    result = ValidationResult()
    iterator = iter(result)
    first = next(iterator)
    second = next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


# LLM-generated content at query #13
#--------------------------

```python
def test_ValidationResult___iter__():
    # Test with value only
    result = ValidationResult(value="test_value")
    value, error = result
    assert value == "test_value"
    assert error is None
    
    # Test with error only
    validation_error = ValidationError(text="Error message", code="custom")
    result = ValidationResult(error=validation_error)
    value, error = result
    assert value is None
    assert error == validation_error
    
    # Test iteration in for loop
    result = ValidationResult(value=42)
    for item in result:
        if item is not None:
            assert item == 42
    
    # Test unpacking with tuple assignment
    result = ValidationResult(value={"key": "value"})
    val, err = result
    assert val == {"key": "value"}
    assert err is None
    
    # Test with None value and error
    result = ValidationResult(value=None)
    value, error = result
    assert value is None
    assert error is None


# LLM-generated content at query #14
#--------------------------

```python
def test_BaseError___str__():
    # Test single message without index
    error = BaseError(text="Invalid value", code="invalid")
    assert str(error) == "Invalid value"
    
    # Test single message with index
    error = BaseError(text="Field is required", code="required", key="username")
    assert str(error) == "{'username': 'Field is required'}"
    
    # Test multiple messages with nested structure
    messages = [
        Message(text="Invalid email", code="invalid", index=["email"]),
        Message(text="Too short", code="min_length", index=["password"]),
    ]
    error = BaseError(messages=messages)
    assert str(error) == "{'email': 'Invalid email', 'password': 'Too short'}"
    
    # Test multiple messages with deeper nesting
    messages = [
        Message(text="Invalid value", code="invalid", index=["users", 0, "email"]),
        Message(text="Required field", code="required", index=["users", 1, "name"]),
    ]
    error = BaseError(messages=messages)
    expected = "{'users': {0: {'email': 'Invalid value'}, 1: {'name': 'Required field'}}}"
    assert str(error) == expected
    
    # Test empty index list (should be treated as root)
    messages = [
        Message(text="Root error", code="root_error", index=[]),
        Message(text="Another error", code="another", index=["field"]),
    ]
    error = BaseError(messages=messages)
    assert str(error) == "{'': 'Root error', 'field': 'Another error'}"
    
    # Test with position information (should not affect string representation)
    position = Position(line_no=1, column_no=5, char_index=4)
    error = BaseError(text="Error with position", code="error", position=position)
    assert str(error) == "Error with position"
    
    # Test mixed index types (string and integer)
    messages = [
        Message(text="Error 1", code="err1", index=["a", 0]),
        Message(text="Error 2", code="err2", index=["b", "key"]),
    ]
    error = BaseError(messages=messages)
    expected = "{'a': {0: 'Error 1'}, 'b': {'key': 'Error 2'}}"
    assert str(error) == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error 2", code="code2", index=["field2"])
    
    error1 = BaseError(messages=[msg1, msg2])
    error2 = BaseError(messages=[msg1, msg2])
    
    assert error1 == error2
    
    # Test equality with different order of messages
    error3 = BaseError(messages=[msg2, msg1])
    assert error1 == error3
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3", index=["field3"])
    error4 = BaseError(messages=[msg1, msg3])
    assert error1 != error4
    
    # Test inequality with different number of messages
    error5 = BaseError(messages=[msg1])
    assert error1 != error5
    
    # Test equality with single message constructor
    error6 = BaseError(text="Single error", code="single", key="field")
    error7 = BaseError(text="Single error", code="single", key="field")
    assert error6 == error7
    
    # Test inequality with single message constructor different text
    error8 = BaseError(text="Different error", code="single", key="field")
    assert error6 != error8
    
    # Test inequality with single message constructor different code
    error9 = BaseError(text="Single error", code="different", key="field")
    assert error6 != error9
    
    # Test inequality with single message constructor different key
    error10 = BaseError(text="Single error", code="single", key="different")
    assert error6 != error10
    
    # Test inequality with different types
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test equality with same messages but different position objects with same values
    pos3 = Position(line_no=1, column_no=2, char_index=3)
    pos4 = Position(line_no=4, column_no=5, char_index=6)
    msg4 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos3, end_position=pos4)
    error11 = BaseError(messages=[msg4, msg2])
    assert error1 == error11
    
    # Test inequality with messages having different positions
    pos5 = Position(line_no=7, column_no=8, char_index=9)
    msg5 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos5, end_position=pos2)
    error12 = BaseError(messages=[msg5, msg2])
    assert error1 != error12
    
    # Test hash equality for equal objects
    assert hash(error1) == hash(error2)
    
    # Test that ValidationError and ParseError also work with equality
    validation_error1 = ValidationError(messages=[msg1, msg2])
    validation_error2 = ValidationError(messages=[msg1, msg2])
    assert validation_error1 == validation_error2
    
    parse_error1 = ParseError(messages=[msg1, msg2])
    parse_error2 = ParseError(messages=[msg1, msg2])
    assert parse_error1 == parse_error2
    
    # Test that different error types are not equal even with same messages
    assert error1 != validation_error1
    assert error1 != parse_error1
    assert validation_error1 != parse_error1


# LLM-generated content at query #16
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    msg4 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    assert msg3 == msg4
    
    # Test equality with start/end positions
    msg5 = Message(text="Error", code="invalid", index=["field"], 
                   start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    msg6 = Message(text="Error", code="invalid", index=["field"], 
                   start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    assert msg5 == msg6
    
    # Test inequality with different text
    msg7 = Message(text="Error 1", code="invalid", key="field")
    msg8 = Message(text="Error 2", code="invalid", key="field")
    assert msg7 != msg8
    
    # Test inequality with different code
    msg9 = Message(text="Error", code="invalid", key="field")
    msg10 = Message(text="Error", code="required", key="field")
    assert msg9 != msg10
    
    # Test inequality with different index
    msg11 = Message(text="Error", code="invalid", index=["field1"])
    msg12 = Message(text="Error", code="invalid", index=["field2"])
    assert msg11 != msg12
    
    # Test inequality with different positions
    msg13 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg14 = Message(text="Error", code="invalid", key="field", position=Position(2, 3, 4))
    assert msg13 != msg14
    
    # Test inequality with different start/end positions
    msg15 = Message(text="Error", code="invalid", index=["field"], 
                    start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    msg16 = Message(text="Error", code="invalid", index=["field"], 
                    start_position=Position(1, 2, 3), end_position=Position(1, 6, 7))
    assert msg15 != msg16
    
    # Test inequality with None positions
    msg17 = Message(text="Error", code="invalid", key="field")
    msg18 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    assert msg17 != msg18
    
    # Test equality with None positions
    msg19 = Message(text="Error", code="invalid", key="field")
    msg20 = Message(text="Error", code="invalid", key="field")
    assert msg19 == msg20
    
    # Test inequality with different type
    msg21 = Message(text="Error", code="invalid", key="field")
    assert msg21 != "not a Message"
    assert msg21 != 123
    assert msg21 != None
    
    # Test equality with default code
    msg22 = Message(text="Error", key="field")
    msg23 = Message(text="Error", code="custom", key="field")
    assert msg22 == msg23
    
    # Test equality with empty index
    msg24 = Message(text="Error", code="invalid")
    msg25 = Message(text="Error", code="invalid", index=[])
    assert msg24 == msg25
    
    # Test equality with position vs start/end position
    msg26 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg27 = Message(text="Error", code="invalid", key="field", 
                    start_position=Position(1, 2, 3), end_position=Position(1, 2, 3))
    assert msg26 == msg27


# LLM-generated content at query #17
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same single message
    error1 = ValidationError(text="Error message", code="custom")
    error2 = ValidationError(text="Error message", code="custom")
    assert error1 == error2
    
    # Test equality with same multiple messages
    msg1 = Message(text="Error 1", code="code1", key="field1")
    msg2 = Message(text="Error 2", code="code2", key="field2")
    error3 = ValidationError(messages=[msg1, msg2])
    error4 = ValidationError(messages=[msg1, msg2])
    assert error3 == error4
    
    # Test inequality with different messages
    error5 = ValidationError(text="Error message 1", code="custom")
    error6 = ValidationError(text="Error message 2", code="custom")
    assert error5 != error6
    
    # Test inequality with different codes
    error7 = ValidationError(text="Error message", code="code1")
    error8 = ValidationError(text="Error message", code="code2")
    assert error7 != error8
    
    # Test inequality with different types
    error9 = ValidationError(text="Error message", code="custom")
    assert error9 != "not an error"
    assert error9 != ParseError(text="Error message", code="custom")
    
    # Test equality with same messages but different order
    msg3 = Message(text="Error 1", code="code1", key="field1")
    msg4 = Message(text="Error 2", code="code2", key="field2")
    error10 = ValidationError(messages=[msg3, msg4])
    error11 = ValidationError(messages=[msg4, msg3])
    assert error10 == error11
    
    # Test equality with same index structure
    msg5 = Message(text="Error", code="code", index=["users", 0, "name"])
    msg6 = Message(text="Error", code="code", index=["users", 0, "name"])
    error12 = ValidationError(messages=[msg5])
    error13 = ValidationError(messages=[msg6])
    assert error12 == error13
    
    # Test inequality with different index structure
    msg7 = Message(text="Error", code="code", index=["users", 0, "name"])
    msg8 = Message(text="Error", code="code", index=["users", 1, "name"])
    error14 = ValidationError(messages=[msg7])
    error15 = ValidationError(messages=[msg8])
    assert error14 != error15
    
    # Test equality with same position
    pos = Position(line_no=1, column_no=5, char_index=10)
    msg9 = Message(text="Error", code="code", position=pos)
    msg10 = Message(text="Error", code="code", position=pos)
    error16 = ValidationError(messages=[msg9])
    error17 = ValidationError(messages=[msg10])
    assert error16 == error17
    
    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=5, char_index=10)
    pos2 = Position(line_no=2, column_no=5, char_index=20)
    msg11 = Message(text="Error", code="code", position=pos1)
    msg12 = Message(text="Error", code="code", position=pos2)
    error18 = ValidationError(messages=[msg11])
    error19 = ValidationError(messages=[msg12])
    assert error18 != error19


# LLM-generated content at query #18
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="custom", index=["field"])
    msg4 = Message(text="Error", code="custom", index=["field"])
    assert msg3 == msg4
    
    # Test equality with position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg5 = Message(text="Error", code="custom", position=pos)
    msg6 = Message(text="Error", code="custom", position=pos)
    assert msg5 == msg6
    
    # Test equality with start/end position
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg7 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    msg8 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    assert msg7 == msg8
    
    # Test inequality with different text
    msg9 = Message(text="Error 1", code="custom")
    msg10 = Message(text="Error 2", code="custom")
    assert not (msg9 == msg10)
    
    # Test inequality with different code
    msg11 = Message(text="Error", code="code1")
    msg12 = Message(text="Error", code="code2")
    assert not (msg11 == msg12)
    
    # Test inequality with different index
    msg13 = Message(text="Error", code="custom", index=["field1"])
    msg14 = Message(text="Error", code="custom", index=["field2"])
    assert not (msg13 == msg14)
    
    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg15 = Message(text="Error", code="custom", position=pos1)
    msg16 = Message(text="Error", code="custom", position=pos2)
    assert not (msg15 == msg16)
    
    # Test inequality with different start position
    start_pos1 = Position(line_no=1, column_no=1, char_index=0)
    start_pos2 = Position(line_no=2, column_no=1, char_index=10)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg17 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos)
    msg18 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos)
    assert not (msg17 == msg18)
    
    # Test inequality with different end position
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos1 = Position(line_no=1, column_no=5, char_index=4)
    end_pos2 = Position(line_no=1, column_no=10, char_index=9)
    msg19 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos1)
    msg20 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos2)
    assert not (msg19 == msg20)
    
    # Test inequality with different types
    msg21 = Message(text="Error", code="custom")
    assert not (msg21 == "not a Message")
    assert not (msg21 == None)
    
    # Test equality with None code (should default to "custom")
    msg22 = Message(text="Error")
    msg23 = Message(text="Error", code="custom")
    assert msg22 == msg23
    
    # Test equality with None index (should default to [])
    msg24 = Message(text="Error", code="custom")
    msg25 = Message(text="Error", code="custom", index=[])
    assert msg24 == msg25
    
    # Test equality with key vs index
    msg26 = Message(text="Error", code="custom", key="field")
    msg27 = Message(text="Error", code="custom", index=["field"])
    assert msg26 == msg27
    
    # Test equality with position vs start/end position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg28 = Message(text="Error", code="custom", position=pos)
    msg29 = Message(text="Error", code="custom", start_position=pos, end_position=pos)
    assert msg28 == msg29


# LLM-generated content at query #19
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2
    assert not (msg1 != msg2)

    # Test equality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test equality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test equality with different index
    msg1 = Message(text="Error", code="custom", index=["users", 0])
    msg2 = Message(text="Error", code="custom", index=["users", 1])
    assert msg1 != msg2

    # Test equality with different position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", code="custom", position=pos1)
    msg2 = Message(text="Error", code="custom", position=pos2)
    assert msg1 != msg2

    # Test equality with different start/end positions
    start1 = Position(line_no=1, column_no=1, char_index=0)
    end1 = Position(line_no=1, column_no=5, char_index=4)
    start2 = Position(line_no=1, column_no=1, char_index=0)
    end2 = Position(line_no=1, column_no=6, char_index=5)
    msg1 = Message(text="Error", code="custom", start_position=start1, end_position=end1)
    msg2 = Message(text="Error", code="custom", start_position=start2, end_position=end2)
    assert msg1 != msg2

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with same start/end positions
    start = Position(line_no=1, column_no=1, char_index=0)
    end = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", code="custom", start_position=start, end_position=end)
    msg2 = Message(text="Error", code="custom", start_position=start, end_position=end)
    assert msg1 == msg2

    # Test equality with key vs index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with different types
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None

    # Test equality with same position object
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", position=pos)
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error", code="custom", index=[])
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


# LLM-generated content at query #20
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error 2", code="code2", index=["field2"])
    
    error1 = BaseError(messages=[msg1, msg2])
    error2 = BaseError(messages=[msg1, msg2])
    assert error1 == error2
    
    # Test equality with different order of messages
    error3 = BaseError(messages=[msg2, msg1])
    assert error1 == error3  # Should be equal regardless of order
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3", index=["field3"])
    error4 = BaseError(messages=[msg1, msg3])
    assert error1 != error4
    
    # Test equality with single message constructor
    error5 = BaseError(text="Single error", code="single")
    error6 = BaseError(text="Single error", code="single")
    assert error5 == error6
    
    # Test inequality with different single messages
    error7 = BaseError(text="Different error", code="single")
    assert error5 != error7
    
    # Test inequality with different types
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test equality with ValidationError subclass
    validation_error1 = ValidationError(messages=[msg1, msg2])
    validation_error2 = ValidationError(messages=[msg1, msg2])
    assert validation_error1 == validation_error2
    
    # Test BaseError not equal to ValidationError with same messages
    assert error1 != validation_error1
    
    # Test equality with ParseError subclass
    parse_error1 = ParseError(messages=[msg1, msg2])
    parse_error2 = ParseError(messages=[msg1, msg2])
    assert parse_error1 == parse_error2
    
    # Test hash equality for same messages
    assert hash(error1) == hash(error2)
    assert hash(error5) == hash(error6)
    
    # Test hash inequality for different messages
    assert hash(error1) != hash(error4)
    assert hash(error5) != hash(error7)


# LLM-generated content at query #21
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert msg1 != msg2

    # Test equality with same position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", position=pos)
    assert msg1 == msg2

    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos2)
    assert msg1 != msg2

    # Test equality with same start/end positions
    start = Position(line_no=1, column_no=1, char_index=0)
    end = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", start_position=start, end_position=end)
    msg2 = Message(text="Error", start_position=start, end_position=end)
    assert msg1 == msg2

    # Test inequality with different start/end positions
    start1 = Position(line_no=1, column_no=1, char_index=0)
    end1 = Position(line_no=1, column_no=5, char_index=4)
    start2 = Position(line_no=2, column_no=1, char_index=10)
    end2 = Position(line_no=2, column_no=5, char_index=14)
    msg1 = Message(text="Error", start_position=start1, end_position=end1)
    msg2 = Message(text="Error", start_position=start2, end_position=end2)
    assert msg1 != msg2

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2

    # Test inequality with different types
    msg = Message(text="Error")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None

    # Test equality with key instead of index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test inequality when one has position and other doesn't
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error")
    assert msg1 != msg2

    # Test equality with complex index
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #22
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error 2", code="code2", index=["field2"])
    
    error1 = BaseError(messages=[msg1, msg2])
    error2 = BaseError(messages=[msg1, msg2])
    
    assert error1 == error2
    
    # Test equality with single message constructor
    error3 = BaseError(text="Single error", code="single", key="field")
    error4 = BaseError(text="Single error", code="single", key="field")
    
    assert error3 == error4
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3")
    error5 = BaseError(messages=[msg1, msg3])
    
    assert error1 != error5
    
    # Test inequality with different message order (should still be equal since we compare sets)
    error6 = BaseError(messages=[msg2, msg1])
    assert error1 == error6  # Order shouldn't matter for equality
    
    # Test inequality with different types
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test inequality with different ValidationError instance but same messages
    validation_error = ValidationError(messages=[msg1, msg2])
    assert error1 == validation_error  # BaseError should equal ValidationError with same messages
    
    # Test inequality with ParseError instance but same messages
    parse_error = ParseError(messages=[msg1, msg2])
    assert error1 == parse_error  # BaseError should equal ParseError with same messages
    
    # Test equality with same messages but different position objects with same values
    pos3 = Position(line_no=1, column_no=2, char_index=3)
    pos4 = Position(line_no=4, column_no=5, char_index=6)
    msg4 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos3, end_position=pos4)
    error7 = BaseError(messages=[msg4, msg2])
    
    assert error1 == error7
    
    # Test inequality with different index
    msg5 = Message(text="Error 1", code="code1", index=["different_field"], start_position=pos1, end_position=pos2)
    error8 = BaseError(messages=[msg5, msg2])
    
    assert error1 != error8
    
    # Test inequality with different code
    msg6 = Message(text="Error 1", code="different_code", index=["field1"], start_position=pos1, end_position=pos2)
    error9 = BaseError(messages=[msg6, msg2])
    
    assert error1 != error9
    
    # Test inequality with different text
    msg7 = Message(text="Different text", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    error10 = BaseError(messages=[msg7, msg2])
    
    assert error1 != error10
    
    # Test equality with empty BaseError (should not happen in practice but test edge case)
    msg8 = Message(text="Test", code="test")
    error11 = BaseError(messages=[msg8])
    error12 = BaseError(messages=[msg8])
    
    assert error11 == error12


# LLM-generated content at query #23
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", position=pos1)
    msg2 = Message(text="Error 2", code="code2", position=pos2)
    
    error1 = ValidationError(messages=[msg1, msg2])
    error2 = ValidationError(messages=[msg1, msg2])
    
    assert error1 == error2
    assert hash(error1) == hash(error2)
    
    # Test equality with different order of messages
    error3 = ValidationError(messages=[msg2, msg1])
    assert error1 != error3
    
    # Test equality with single message constructor
    error4 = ValidationError(text="Single error", code="single")
    error5 = ValidationError(text="Single error", code="single")
    
    assert error4 == error5
    
    # Test inequality with different text
    error6 = ValidationError(text="Different error", code="single")
    assert error4 != error6
    
    # Test inequality with different code
    error7 = ValidationError(text="Single error", code="different")
    assert error4 != error7
    
    # Test inequality with different type
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3")
    error8 = ValidationError(messages=[msg1, msg3])
    assert error1 != error8
    
    # Test equality with same messages but different constructor
    error9 = ValidationError(text="Error 1", code="code1", position=pos1)
    error10 = ValidationError(messages=[msg1])
    assert error9 == error10
    
    # Test equality with ParseError (subclass)
    parse_error1 = ParseError(messages=[msg1, msg2])
    parse_error2 = ParseError(messages=[msg1, msg2])
    assert parse_error1 == parse_error2
    
    # Test inequality between ValidationError and ParseError with same messages
    assert error1 != parse_error1
    
    # Test equality with messages having index
    msg_with_index = Message(text="Indexed error", code="indexed", index=["key1", 0])
    error11 = ValidationError(messages=[msg_with_index])
    error12 = ValidationError(messages=[msg_with_index])
    assert error11 == error12
    
    # Test inequality with different index
    msg_diff_index = Message(text="Indexed error", code="indexed", index=["key1", 1])
    error13 = ValidationError(messages=[msg_diff_index])
    assert error11 != error13


# LLM-generated content at query #24
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error 2", code="code2", index=["field2"])
    
    error1 = BaseError(messages=[msg1, msg2])
    error2 = BaseError(messages=[msg1, msg2])
    
    assert error1 == error2
    
    # Test equality with different order of messages
    error3 = BaseError(messages=[msg2, msg1])
    assert error1 == error3
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3", index=["field3"])
    error4 = BaseError(messages=[msg1, msg3])
    assert error1 != error4
    
    # Test inequality with different number of messages
    error5 = BaseError(messages=[msg1])
    assert error1 != error5
    
    # Test equality with single message constructor
    error6 = BaseError(text="Single error", code="single", key="field")
    error7 = BaseError(messages=[Message(text="Single error", code="single", key="field")])
    assert error6 == error7
    
    # Test inequality with different object type
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test equality with ValidationError subclass
    validation_error = ValidationError(messages=[msg1, msg2])
    assert error1 == validation_error
    
    # Test equality with ParseError subclass
    parse_error = ParseError(messages=[msg1, msg2])
    assert error1 == parse_error
    
    # Test hash equality for equal objects
    assert hash(error1) == hash(error2)
    
    # Test with messages having same hash but different content
    msg4 = Message(text="Error", code="code", index=["field"])
    msg5 = Message(text="Different", code="code", index=["field"])
    
    error8 = BaseError(messages=[msg4])
    error9 = BaseError(messages=[msg5])
    
    assert error8 != error9


# LLM-generated content at query #25
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same single message
    error1 = ValidationError(text="Error message", code="custom")
    error2 = ValidationError(text="Error message", code="custom")
    assert error1 == error2
    assert hash(error1) == hash(error2)

    # Test equality with same multiple messages
    messages = [
        Message(text="Error 1", code="code1", index=["field1"]),
        Message(text="Error 2", code="code2", index=["field2"])
    ]
    error3 = ValidationError(messages=messages)
    error4 = ValidationError(messages=messages)
    assert error3 == error4
    assert hash(error3) == hash(error4)

    # Test inequality with different messages
    error5 = ValidationError(text="Error message", code="custom")
    error6 = ValidationError(text="Different message", code="custom")
    assert error5 != error6
    assert hash(error5) != hash(error6)

    # Test inequality with different number of messages
    messages1 = [Message(text="Error 1", code="code1")]
    messages2 = [
        Message(text="Error 1", code="code1"),
        Message(text="Error 2", code="code2")
    ]
    error7 = ValidationError(messages=messages1)
    error8 = ValidationError(messages=messages2)
    assert error7 != error8
    assert hash(error7) != hash(error8)

    # Test inequality with same text but different code
    error9 = ValidationError(text="Error message", code="code1")
    error10 = ValidationError(text="Error message", code="code2")
    assert error9 != error10
    assert hash(error9) != hash(error10)

    # Test inequality with different types
    error11 = ValidationError(text="Error message")
    assert error11 != "not an error"
    assert error11 != 123
    assert error11 != None

    # Test equality with same messages but different order
    messages3 = [
        Message(text="Error 1", code="code1", index=["field1"]),
        Message(text="Error 2", code="code2", index=["field2"])
    ]
    messages4 = [
        Message(text="Error 2", code="code2", index=["field2"]),
        Message(text="Error 1", code="code1", index=["field1"])
    ]
    error12 = ValidationError(messages=messages3)
    error13 = ValidationError(messages=messages4)
    assert error12 == error13
    assert hash(error12) == hash(error13)

    # Test equality with same messages including positions
    pos = Position(line_no=1, column_no=1, char_index=0)
    messages5 = [Message(text="Error", code="code", position=pos)]
    messages6 = [Message(text="Error", code="code", position=pos)]
    error14 = ValidationError(messages=messages5)
    error15 = ValidationError(messages=messages6)
    assert error14 == error15
    assert hash(error14) == hash(error15)

    # Test inequality with same messages but different positions
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    messages7 = [Message(text="Error", code="code", position=pos1)]
    messages8 = [Message(text="Error", code="code", position=pos2)]
    error16 = ValidationError(messages=messages7)
    error17 = ValidationError(messages=messages8)
    assert error16 == error17  # Positions don't affect equality for BaseError
    assert hash(error16) == hash(error17)  # Positions don't affect hash for Message

    # Test ParseError equality
    parse_error1 = ParseError(text="Parse error", code="parse_error")
    parse_error2 = ParseError(text="Parse error", code="parse_error")
    assert parse_error1 == parse_error2
    assert hash(parse_error1) == hash(parse_error2)

    # Test ValidationError equality
    validation_error1 = ValidationError(text="Validation error", code="validation")
    validation_error2 = ValidationError(text="Validation error", code="validation")
    assert validation_error1 == validation_error2
    assert hash(validation_error1) == hash(validation_error2)

    # Test that different BaseError subclasses are not equal
    parse_error3 = ParseError(text="Error", code="code")
    validation_error3 = ValidationError(text="Error", code="code")
    assert parse_error3 != validation_error3
    assert hash(parse_error3) != hash(validation_error3)


# LLM-generated content at query #26
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with same values including positions
    pos = Position(1, 1, 0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", position=pos)
    assert msg1 == msg2
    
    # Test equality with same values including start/end positions
    start_pos = Position(1, 1, 0)
    end_pos = Position(1, 5, 4)
    msg1 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2
    
    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)
    
    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)
    
    # Test inequality with different index
    msg1 = Message(text="Error", index=["field1"])
    msg2 = Message(text="Error", index=["field2"])
    assert not (msg1 == msg2)
    
    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert not (msg1 == msg2)
    
    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert not (msg1 == msg2)
    
    # Test inequality with different type
    msg = Message(text="Error")
    assert not (msg == "not a Message")
    
    # Test equality with None index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2
    
    # Test equality with key parameter (should convert to index)
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2
    
    # Test equality with position vs start/end position
    pos = Position(1, 1, 0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    assert msg1 == msg2


# LLM-generated content at query #27
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom", key="field")
    msg2 = Message(text="Error 2", code="custom", key="field")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1", key="field")
    msg2 = Message(text="Error", code="code2", key="field")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field", 0])
    msg2 = Message(text="Error", code="custom", index=["field", 1])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 0))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(1, 2, 0))
    assert msg1 != msg2

    # Test equality with index instead of key
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test hash consistency with equality
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert hash(msg1) == hash(msg2)
    assert msg1 == msg2


# LLM-generated content at query #28
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field"])
    msg2 = Message(text="Error", index=["other"])
    assert msg1 != msg2

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality using position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg1 == msg2

    # Test inequality with different types
    msg = Message(text="Error")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with None code (defaults to "custom")
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2


# LLM-generated content at query #29
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field1"])
    msg2 = Message(text="Error", index=["field2"])
    assert not (msg1 == msg2)

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert not (msg1 == msg2)

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert not (msg1 == msg2)

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality with position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg1 == msg2

    # Test inequality with different types
    msg = Message(text="Error")
    assert not (msg == "not a Message")
    assert not (msg == 123)
    assert not (msg == None)

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test reflexive property
    msg = Message(text="Error", code="max_length", index=["users", 0, "name"])
    assert msg == msg

    # Test symmetric property
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos1)
    assert (msg1 == msg2) == (msg2 == msg1)


# LLM-generated content at query #30
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with None positions
    msg3 = Message(text="Error", code="custom")
    msg4 = Message(text="Error", code="custom")
    assert msg3 == msg4
    
    # Test equality with same position
    pos = Position(1, 2, 3)
    msg5 = Message(text="Error", position=pos)
    msg6 = Message(text="Error", position=pos)
    assert msg5 == msg6
    
    # Test equality with separate start/end positions
    start_pos = Position(1, 2, 3)
    end_pos = Position(1, 5, 6)
    msg7 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    msg8 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    assert msg7 == msg8
    
    # Test inequality with different text
    msg9 = Message(text="Error 1")
    msg10 = Message(text="Error 2")
    assert not (msg9 == msg10)
    
    # Test inequality with different code
    msg11 = Message(text="Error", code="max_length")
    msg12 = Message(text="Error", code="min_length")
    assert not (msg11 == msg12)
    
    # Test inequality with different index
    msg13 = Message(text="Error", index=["field1"])
    msg14 = Message(text="Error", index=["field2"])
    assert not (msg13 == msg14)
    
    # Test inequality with different start_position
    pos1 = Position(1, 2, 3)
    pos2 = Position(2, 3, 4)
    msg15 = Message(text="Error", start_position=pos1)
    msg16 = Message(text="Error", start_position=pos2)
    assert not (msg15 == msg16)
    
    # Test inequality with different end_position
    msg17 = Message(text="Error", end_position=pos1)
    msg18 = Message(text="Error", end_position=pos2)
    assert not (msg17 == msg18)
    
    # Test inequality with different types
    msg19 = Message(text="Error")
    assert not (msg19 == "not a Message")
    assert not (msg19 == None)
    
    # Test equality with key instead of index
    msg20 = Message(text="Error", key="field")
    msg21 = Message(text="Error", index=["field"])
    assert msg20 == msg21
    
    # Test equality with position instead of start/end
    msg22 = Message(text="Error", position=pos1)
    msg23 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg22 == msg23
    
    # Test hash equality for equal messages
    msg24 = Message(text="Error", code="custom", index=["field"])
    msg25 = Message(text="Error", code="custom", index=["field"])
    assert hash(msg24) == hash(msg25)
    
    # Test hash inequality for different messages
    msg26 = Message(text="Error", code="custom", index=["field1"])
    msg27 = Message(text="Error", code="custom", index=["field2"])
    assert hash(msg26) != hash(msg27)


# LLM-generated content at query #31
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with None positions
    msg3 = Message(text="Error", code="custom")
    msg4 = Message(text="Error", code="custom")
    assert msg3 == msg4

    # Test equality with same position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg5 = Message(text="Error", position=pos)
    msg6 = Message(text="Error", position=pos)
    assert msg5 == msg6

    # Test equality with same start and end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg7 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    msg8 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    assert msg7 == msg8

    # Test inequality with different text
    msg9 = Message(text="Error 1")
    msg10 = Message(text="Error 2")
    assert not (msg9 == msg10)

    # Test inequality with different code
    msg11 = Message(text="Error", code="code1")
    msg12 = Message(text="Error", code="code2")
    assert not (msg11 == msg12)

    # Test inequality with different index
    msg13 = Message(text="Error", index=["field1"])
    msg14 = Message(text="Error", index=["field2"])
    assert not (msg13 == msg14)

    # Test inequality with different start_position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg15 = Message(text="Error", start_position=pos1)
    msg16 = Message(text="Error", start_position=pos2)
    assert not (msg15 == msg16)

    # Test inequality with different end_position
    msg17 = Message(text="Error", end_position=pos1)
    msg18 = Message(text="Error", end_position=pos2)
    assert not (msg17 == msg18)

    # Test inequality with different type
    msg19 = Message(text="Error")
    assert not (msg19 == "not a Message")

    # Test equality with key instead of index
    msg20 = Message(text="Error", key="field")
    msg21 = Message(text="Error", index=["field"])
    assert msg20 == msg21

    # Test equality with position instead of start/end
    msg22 = Message(text="Error", position=pos1)
    msg23 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg22 == msg23

    # Test inequality when one has position and other has separate start/end
    msg24 = Message(text="Error", position=pos1)
    msg25 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert not (msg24 == msg25)


# LLM-generated content at query #32
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    assert msg1 == msg2
    
    # Test inequality with different text
    msg3 = Message(text="Different", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    assert msg1 != msg3
    
    # Test inequality with different code
    msg4 = Message(text="Error", code="different", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    assert msg1 != msg4
    
    # Test inequality with different index
    msg5 = Message(text="Error", code="custom", index=[1], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    assert msg1 != msg5
    
    # Test inequality with different start_position
    msg6 = Message(text="Error", code="custom", index=[0], start_position=Position(2, 1, 0), end_position=Position(1, 5, 4))
    assert msg1 != msg6
    
    # Test inequality with different end_position
    msg7 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(2, 5, 4))
    assert msg1 != msg7
    
    # Test equality with None positions
    msg8 = Message(text="Error", code="custom", index=[0])
    msg9 = Message(text="Error", code="custom", index=[0])
    assert msg8 == msg9
    
    # Test inequality with mixed None positions
    msg10 = Message(text="Error", code="custom", index=[0])
    assert msg1 != msg10
    
    # Test equality with position parameter (instead of start/end)
    msg11 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg12 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    assert msg11 == msg12
    
    # Test inequality with different position
    msg13 = Message(text="Error", code="custom", index=[0], position=Position(2, 1, 0))
    assert msg11 != msg13
    
    # Test equality with key parameter
    msg14 = Message(text="Error", code="custom", key="field")
    msg15 = Message(text="Error", code="custom", key="field")
    assert msg14 == msg15
    
    # Test inequality with different key
    msg16 = Message(text="Error", code="custom", key="different")
    assert msg14 != msg16
    
    # Test equality with empty index
    msg17 = Message(text="Error", code="custom")
    msg18 = Message(text="Error", code="custom")
    assert msg17 == msg18
    
    # Test comparison with non-Message object
    assert msg1 != "not a message"
    assert msg1 != 123
    assert msg1 != None
    
    # Test comparison with same object
    assert msg1 == msg1


# LLM-generated content at query #33
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(1, 5, 4))
    msg2 = Message(text="Error", code="custom", index=[0], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(1, 5, 4))
    assert msg1 == msg2
    
    # Test inequality with different text
    msg3 = Message(text="Different", code="custom", index=[0], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(1, 5, 4))
    assert msg1 != msg3
    
    # Test inequality with different code
    msg4 = Message(text="Error", code="max_length", index=[0], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(1, 5, 4))
    assert msg1 != msg4
    
    # Test inequality with different index
    msg5 = Message(text="Error", code="custom", index=[1], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(1, 5, 4))
    assert msg1 != msg5
    
    # Test inequality with different start_position
    msg6 = Message(text="Error", code="custom", index=[0], 
                   start_position=Position(2, 1, 0), 
                   end_position=Position(1, 5, 4))
    assert msg1 != msg6
    
    # Test inequality with different end_position
    msg7 = Message(text="Error", code="custom", index=[0], 
                   start_position=Position(1, 1, 0), 
                   end_position=Position(2, 5, 4))
    assert msg1 != msg7
    
    # Test equality with None positions
    msg8 = Message(text="Error", code="custom", index=[0])
    msg9 = Message(text="Error", code="custom", index=[0])
    assert msg8 == msg9
    
    # Test inequality with None vs not None positions
    assert msg1 != msg8
    
    # Test equality with position parameter (start_position == end_position)
    msg10 = Message(text="Error", code="custom", index=[0], 
                    position=Position(1, 1, 0))
    msg11 = Message(text="Error", code="custom", index=[0], 
                    start_position=Position(1, 1, 0), 
                    end_position=Position(1, 1, 0))
    assert msg10 == msg11
    
    # Test equality with key parameter (converts to index)
    msg12 = Message(text="Error", code="custom", key="field")
    msg13 = Message(text="Error", code="custom", index=["field"])
    assert msg12 == msg13
    
    # Test inequality with different object type
    assert msg1 != "not a Message"
    assert msg1 != 123
    assert msg1 != None
    
    # Test equality with empty index
    msg14 = Message(text="Error", code="custom")
    msg15 = Message(text="Error", code="custom", index=[])
    assert msg14 == msg15
    
    # Test equality with complex index
    msg16 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg17 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg16 == msg17


# LLM-generated content at query #34
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", index=[0])
    msg2 = Message(text="Error", index=[1])
    assert msg1 != msg2

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 2, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality with position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error", index=[])
    msg2 = Message(text="Error")
    assert msg1 == msg2


# LLM-generated content at query #35
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    assert not (msg1 != msg2)

    # Test equality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test equality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test equality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test equality with different positions
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", code="custom", position=pos1)
    msg2 = Message(text="Error", code="custom", position=pos2)
    assert msg1 != msg2

    # Test equality with same positions
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", position=pos)
    assert msg1 == msg2

    # Test equality with start/end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2

    # Test equality with different start/end positions
    start_pos1 = Position(line_no=1, column_no=1, char_index=0)
    end_pos1 = Position(line_no=1, column_no=5, char_index=4)
    start_pos2 = Position(line_no=2, column_no=1, char_index=10)
    end_pos2 = Position(line_no=2, column_no=5, char_index=14)
    msg1 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos1)
    msg2 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos2)
    assert msg1 != msg2

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with different types
    msg = Message(text="Error", code="custom")
    assert msg != "not a message"
    assert msg != 123
    assert msg != None

    # Test equality with same key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with different key/index combinations
    msg1 = Message(text="Error", code="custom", key="field1")
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test equality with empty index
    msg1 = Message(text="Error", code="custom", index=[])
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with nested index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2

    # Test equality with different nested index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 1, "name"])
    assert msg1 != msg2


# LLM-generated content at query #36
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    pos1 = Position(1, 2, 3)
    pos2 = Position(4, 5, 6)
    
    msg1 = Message(
        text="Error message",
        code="max_length",
        key="username",
        start_position=pos1,
        end_position=pos2
    )
    
    msg2 = Message(
        text="Error message",
        code="max_length",
        key="username",
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(
        text="Error message",
        code="max_length",
        index=["users", 0, "username"],
        start_position=pos1,
        end_position=pos2
    )
    
    msg4 = Message(
        text="Error message",
        code="max_length",
        index=["users", 0, "username"],
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg3 == msg4
    
    # Test equality with position parameter
    msg5 = Message(
        text="Error message",
        code="max_length",
        key="username",
        position=pos1
    )
    
    msg6 = Message(
        text="Error message",
        code="max_length",
        key="username",
        position=pos1
    )
    
    assert msg5 == msg6
    
    # Test inequality with different text
    msg7 = Message(
        text="Error message 1",
        code="max_length",
        key="username"
    )
    
    msg8 = Message(
        text="Error message 2",
        code="max_length",
        key="username"
    )
    
    assert msg7 != msg8
    
    # Test inequality with different code
    msg9 = Message(
        text="Error message",
        code="max_length",
        key="username"
    )
    
    msg10 = Message(
        text="Error message",
        code="min_length",
        key="username"
    )
    
    assert msg9 != msg10
    
    # Test inequality with different index
    msg11 = Message(
        text="Error message",
        code="max_length",
        index=["users", 0, "username"]
    )
    
    msg12 = Message(
        text="Error message",
        code="max_length",
        index=["users", 1, "username"]
    )
    
    assert msg11 != msg12
    
    # Test inequality with different positions
    msg13 = Message(
        text="Error message",
        code="max_length",
        key="username",
        start_position=pos1,
        end_position=pos2
    )
    
    msg14 = Message(
        text="Error message",
        code="max_length",
        key="username",
        start_position=pos2,
        end_position=pos1
    )
    
    assert msg13 != msg14
    
    # Test inequality with different types
    msg15 = Message(text="Error message")
    assert msg15 != "not a Message"
    assert msg15 != 123
    assert msg15 != None
    
    # Test equality with default code
    msg16 = Message(text="Error message")
    msg17 = Message(text="Error message", code="custom")
    assert msg16 == msg17
    
    # Test equality with empty index
    msg18 = Message(text="Error message", key="username")
    msg19 = Message(text="Error message", index=["username"])
    assert msg18 == msg19
    
    # Test equality with None positions
    msg20 = Message(text="Error message", code="max_length")
    msg21 = Message(text="Error message", code="max_length")
    assert msg20 == msg21


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_BaseError___eq__():
    # Test equality with same messages
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(text="Error 1", code="code1", index=["field1"], start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error 2", code="code2", index=["field2"])
    
    error1 = BaseError(messages=[msg1, msg2])
    error2 = BaseError(messages=[msg1, msg2])
    
    assert error1 == error2
    assert hash(error1) == hash(error2)
    
    # Test equality with different order of messages (should still be equal)
    error3 = BaseError(messages=[msg2, msg1])
    assert error1 == error3
    assert hash(error1) == hash(error3)
    
    # Test inequality with different messages
    msg3 = Message(text="Error 3", code="code3", index=["field3"])
    error4 = BaseError(messages=[msg1, msg3])
    
    assert error1 != error4
    assert hash(error1) != hash(error4)
    
    # Test inequality with different number of messages
    error5 = BaseError(messages=[msg1])
    assert error1 != error5
    
    # Test equality with single message constructor
    error6 = BaseError(text="Single error", code="single", key="field")
    error7 = BaseError(text="Single error", code="single", key="field")
    
    assert error6 == error7
    assert hash(error6) == hash(error7)
    
    # Test inequality with different single message
    error8 = BaseError(text="Different error", code="single", key="field")
    assert error6 != error8
    
    # Test inequality with different types
    assert error1 != "not an error"
    assert error1 != 123
    assert error1 != None
    
    # Test equality with ValidationError subclass
    validation_error1 = ValidationError(messages=[msg1, msg2])
    validation_error2 = ValidationError(messages=[msg1, msg2])
    
    assert validation_error1 == validation_error2
    assert hash(validation_error1) == hash(validation_error2)
    
    # Test that BaseError and ValidationError are not equal even with same messages
    # because isinstance check in BaseError.__eq__ checks for ValidationError
    base_error = BaseError(messages=[msg1, msg2])
    assert base_error != validation_error1
    
    # Test equality with ParseError subclass
    parse_error1 = ParseError(messages=[msg1, msg2])
    parse_error2 = ParseError(messages=[msg1, msg2])
    
    assert parse_error1 == parse_error2
    assert hash(parse_error1) == hash(parse_error2)
    
    # Test with messages having same hash but different content
    # (should not happen due to hash implementation, but testing edge case)
    msg_same_hash1 = Message(text="A", code="code", index=["field"])
    msg_same_hash2 = Message(text="B", code="code", index=["field"])
    
    # These have same hash because hash only uses code and index
    assert hash(msg_same_hash1) == hash(msg_same_hash2)
    
    error_hash1 = BaseError(messages=[msg_same_hash1])
    error_hash2 = BaseError(messages=[msg_same_hash2])
    
    # Should not be equal despite same hash because text differs
    assert error_hash1 != error_hash2
    
    # Test with empty index messages
    msg_no_index1 = Message(text="Global error", code="global")
    msg_no_index2 = Message(text="Global error", code="global")
    
    error_no_index1 = BaseError(messages=[msg_no_index1])
    error_no_index2 = BaseError(messages=[msg_no_index2])
    
    assert error_no_index1 == error_no_index2
    
    # Test with position variations
    msg_pos1 = Message(text="Error", code="code", position=pos1)
    msg_pos2 = Message(text="Error", code="code", position=pos1)
    msg_pos3 = Message(text="Error", code="code", position=pos2)
    
    error_pos1 = BaseError(messages=[msg_pos1])
    error_pos2 = BaseError(messages=[msg_pos2])
    error_pos3 = BaseError(messages=[msg_pos3])
    
    assert error_pos1 == error_pos2
    assert error_pos1 != error_pos3


# LLM-generated content at query #2
#--------------------------

```python
def test_BaseError():
    # Test initialization with single message using text parameter
    error1 = BaseError(text="Field is required", code="required", key="username")
    assert error1.text == "Field is required"
    assert error1.code == "required"
    assert error1.index == ["username"]
    assert len(error1._messages) == 1
    assert error1._messages[0].text == "Field is required"
    assert error1._messages[0].code == "required"
    assert error1._messages[0].index == ["username"]
    
    # Test initialization with single message using position
    position = Position(line_no=1, column_no=5, char_index=4)
    error2 = BaseError(text="Invalid format", code="format", position=position)
    assert error2._messages[0].start_position == position
    assert error2._messages[0].end_position == position
    
    # Test initialization with multiple messages
    messages = [
        Message(text="Too short", code="min_length", key="password"),
        Message(text="Missing uppercase", code="pattern", key="password")
    ]
    error3 = BaseError(messages=messages)
    assert len(error3._messages) == 2
    assert error3._messages == messages
    
    # Test that mixed initialization raises AssertionError
    try:
        BaseError(text="Test", messages=messages)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test empty messages list raises AssertionError
    try:
        BaseError(messages=[])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test that _message_dict is populated correctly for single message
    error4 = BaseError(text="Error", key="field")
    assert error4._message_dict == {"field": "Error"}
    
    # Test that _message_dict is populated correctly for nested messages
    messages_nested = [
        Message(text="Error 1", index=["users", 0, "name"]),
        Message(text="Error 2", index=["users", 0, "email"])
    ]
    error5 = BaseError(messages=messages_nested)
    assert error5._message_dict == {
        "users": {
            0: {
                "name": "Error 1",
                "email": "Error 2"
            }
        }
    }
    
    # Test that _message_dict handles empty index correctly
    error6 = BaseError(text="Global error")
    assert error6._message_dict == {"": "Global error"}
    
    # Test mapping interface
    error7 = BaseError(text="Error", key="test")
    assert len(error7) == 1
    assert list(error7) == ["test"]
    assert error7["test"] == "Error"
    
    # Test equality
    error8a = BaseError(text="Same", key="key")
    error8b = BaseError(text="Same", key="key")
    error8c = BaseError(text="Different", key="key")
    assert error8a == error8b
    assert error8a != error8c
    
    # Test hash
    error9a = BaseError(text="Test", key="field")
    error9b = BaseError(text="Test", key="field")
    assert hash(error9a) == hash(error9b)
    
    # Test messages() method without prefix
    error10 = BaseError(text="Message", key="field")
    messages10 = error10.messages()
    assert len(messages10) == 1
    assert messages10[0].text == "Message"
    assert messages10[0].index == ["field"]
    
    # Test messages() method with prefix
    messages11 = error10.messages(add_prefix="parent")
    assert len(messages11) == 1
    assert messages11[0].index == ["parent", "field"]
    
    # Test string representation for single message without index
    error12 = BaseError(text="Simple error")
    assert str(error12) == "Simple error"
    
    # Test string representation for multiple messages
    error13 = BaseError(messages=[
        Message(text="Error 1", key="field1"),
        Message(text="Error 2", key="field2")
    ])
    assert str(error13) == "{'field1': 'Error 1', 'field2': 'Error 2'}"
    
    # Test repr for single message without index
    error14 = BaseError(text="Error", code="custom")
    assert repr(error14) == "BaseError(text='Error', code='custom')"
    
    # Test repr for multiple messages
    error15 = BaseError(messages=[
        Message(text="Error 1", key="field1"),
        Message(text="Error 2", key="field2")
    ])
    assert "BaseError([" in repr(error15)
    
    # Test that BaseError is a subclass of Exception and Mapping
    assert issubclass(BaseError, Exception)
    assert issubclass(BaseError, Mapping)
    
    # Test iteration
    error16 = BaseError(messages=[
        Message(text="A", key="a"),
        Message(text="B", key="b")
    ])
    keys = list(error16)
    assert set(keys) == {"a", "b"}


# LLM-generated content at query #3
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert not (msg1 == msg2)
    
    # Test equality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)
    
    # Test equality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert not (msg1 == msg2)
    
    # Test equality with different positions
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", code="custom", position=pos1)
    msg2 = Message(text="Error", code="custom", position=pos2)
    assert not (msg1 == msg2)
    
    # Test equality with same positions
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", position=pos)
    assert msg1 == msg2
    
    # Test equality with start/end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2
    
    # Test equality with different start/end positions
    start_pos1 = Position(line_no=1, column_no=1, char_index=0)
    end_pos1 = Position(line_no=1, column_no=5, char_index=4)
    start_pos2 = Position(line_no=2, column_no=1, char_index=10)
    end_pos2 = Position(line_no=2, column_no=5, char_index=14)
    msg1 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos1)
    msg2 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos2)
    assert not (msg1 == msg2)
    
    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2
    
    # Test equality with mixed position types
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", start_position=pos, end_position=pos)
    assert msg1 == msg2
    
    # Test equality with different object type
    msg = Message(text="Error", code="custom")
    assert not (msg == "not a message")
    
    # Test equality with key parameter
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with empty index
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom", index=[])
    assert msg1 == msg2


# LLM-generated content at query #4
#--------------------------

```python
def test_BaseError___str__():
    # Test single message without index
    error1 = ValidationError(text="Invalid value", code="invalid")
    assert str(error1) == "Invalid value"
    
    # Test single message with index
    error2 = ValidationError(text="Field required", code="required", key="username")
    assert str(error2) == "{'username': 'Field required'}"
    
    # Test multiple messages with different indices
    messages = [
        Message(text="Field required", code="required", index=["username"]),
        Message(text="Too short", code="min_length", index=["password"]),
    ]
    error3 = ValidationError(messages=messages)
    assert str(error3) == "{'username': 'Field required', 'password': 'Too short'}"
    
    # Test nested messages
    messages = [
        Message(text="Invalid email", code="invalid", index=["users", 0, "email"]),
        Message(text="Too long", code="max_length", index=["users", 1, "name"]),
    ]
    error4 = ValidationError(messages=messages)
    expected = "{'users': {0: {'email': 'Invalid email'}, 1: {'name': 'Too long'}}}"
    assert str(error4) == expected
    
    # Test mixed nested and top-level messages
    messages = [
        Message(text="Invalid format", code="invalid", index=[]),
        Message(text="Missing field", code="required", index=["profile", "age"]),
    ]
    error5 = ValidationError(messages=messages)
    assert str(error5) == "{'': 'Invalid format', 'profile': {'age': 'Missing field'}}"
    
    # Test ParseError (subclass of BaseError)
    error6 = ParseError(text="Invalid JSON", code="parse_error")
    assert str(error6) == "Invalid JSON"
    
    # Test with position information (should not affect string representation)
    position = Position(line_no=1, column_no=5, char_index=4)
    error7 = ValidationError(
        text="Syntax error", 
        code="syntax", 
        key="query",
        position=position
    )
    assert str(error7) == "{'query': 'Syntax error'}"


# LLM-generated content at query #5
#--------------------------

```python
def test_BaseError___repr__():
    # Test single message without index
    error1 = ValidationError(text="Invalid value", code="invalid")
    assert repr(error1) == "ValidationError(text='Invalid value', code='invalid')"
    
    # Test single message with index
    error2 = ValidationError(text="Field required", code="required", key="username")
    assert repr(error2) == "ValidationError([Message(text='Field required', code='required', index=['username'])])"
    
    # Test multiple messages
    messages = [
        Message(text="Too short", code="min_length", index=["username"]),
        Message(text="Invalid format", code="format", index=["email"])
    ]
    error3 = ValidationError(messages=messages)
    assert repr(error3) == f"ValidationError({messages!r})"
    
    # Test with position
    pos = Position(line_no=1, column_no=5, char_index=4)
    error4 = ValidationError(text="Syntax error", code="syntax", position=pos)
    assert repr(error4) == f"ValidationError(text='Syntax error', code='syntax')"
    
    # Test ParseError with single message
    error5 = ParseError(text="Invalid JSON", code="invalid_json")
    assert repr(error5) == "ParseError(text='Invalid JSON', code='invalid_json')"
    
    # Test with nested index
    messages2 = [
        Message(text="Invalid", code="invalid", index=["users", 0, "name"])
    ]
    error6 = ValidationError(messages=messages2)
    assert repr(error6) == f"ValidationError({messages2!r})"
    
    # Test empty messages list should raise assertion error
    try:
        ValidationError(messages=[])
    except AssertionError:
        pass
    else:
        assert False, "Should raise AssertionError for empty messages"


# LLM-generated content at query #6
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field1"])
    msg2 = Message(text="Error", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality with position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos1)
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None

    # Test equality with key parameter (should convert to index)
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", start_position=None, end_position=None)
    assert msg1 == msg2


# LLM-generated content at query #7
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert not (msg1 == msg2)

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert not (msg1 == msg2)

    # Test inequality with different start_position
    msg1 = Message(text="Error", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", start_position=Position(2, 1, 10))
    assert not (msg1 == msg2)

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=Position(1, 5, 4))
    msg2 = Message(text="Error", end_position=Position(1, 10, 9))
    assert not (msg1 == msg2)

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", position=Position(1, 1, 0))
    msg2 = Message(text="Error", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert not (msg == "not a message")

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test inequality when one has position and other doesn't
    msg1 = Message(text="Error", position=Position(1, 1, 0))
    msg2 = Message(text="Error")
    assert not (msg1 == msg2)

    # Test equality with complex index
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #8
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    pos1 = Position(line_no=1, column_no=2, char_index=3)
    pos2 = Position(line_no=4, column_no=5, char_index=6)
    
    msg1 = Message(
        text="Error message",
        code="custom",
        index=["field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    msg2 = Message(
        text="Error message",
        code="custom",
        index=["field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg1 == msg2
    assert not (msg1 != msg2)
    
    # Test inequality with different text
    msg3 = Message(
        text="Different message",
        code="custom",
        index=["field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg1 != msg3
    
    # Test inequality with different code
    msg4 = Message(
        text="Error message",
        code="different_code",
        index=["field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg1 != msg4
    
    # Test inequality with different index
    msg5 = Message(
        text="Error message",
        code="custom",
        index=["different_field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg1 != msg5
    
    # Test inequality with different start_position
    pos3 = Position(line_no=7, column_no=8, char_index=9)
    msg6 = Message(
        text="Error message",
        code="custom",
        index=["field", 0],
        start_position=pos3,
        end_position=pos2
    )
    
    assert msg1 != msg6
    
    # Test inequality with different end_position
    msg7 = Message(
        text="Error message",
        code="custom",
        index=["field", 0],
        start_position=pos1,
        end_position=pos3
    )
    
    assert msg1 != msg7
    
    # Test equality with None positions
    msg8 = Message(text="Error message", code="custom")
    msg9 = Message(text="Error message", code="custom")
    
    assert msg8 == msg9
    
    # Test inequality with mixed None positions
    msg10 = Message(
        text="Error message",
        code="custom",
        start_position=pos1,
        end_position=pos2
    )
    
    assert msg8 != msg10
    
    # Test equality with key instead of index
    msg11 = Message(text="Error message", key="field")
    msg12 = Message(text="Error message", key="field")
    
    assert msg11 == msg12
    
    # Test inequality with different key
    msg13 = Message(text="Error message", key="different_field")
    
    assert msg11 != msg13
    
    # Test equality with position parameter
    msg14 = Message(text="Error message", position=pos1)
    msg15 = Message(text="Error message", position=pos1)
    
    assert msg14 == msg15
    
    # Test inequality with different position
    msg16 = Message(text="Error message", position=pos2)
    
    assert msg14 != msg16
    
    # Test equality with same position for start and end
    msg17 = Message(
        text="Error message",
        start_position=pos1,
        end_position=pos1
    )
    msg18 = Message(text="Error message", position=pos1)
    
    assert msg17 == msg18
    
    # Test comparison with non-Message object
    assert msg1 != "not a message"
    assert msg1 != 123
    assert msg1 != None
    
    # Test comparison with Message subclass (should still be equal if attributes match)
    class SubMessage(Message):
        pass
    
    sub_msg = SubMessage(
        text="Error message",
        code="custom",
        index=["field", 0],
        start_position=pos1,
        end_position=pos2
    )
    
    # This should be True because isinstance(other, Message) returns True
    assert msg1 == sub_msg


# LLM-generated content at query #9
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with same values including positions
    pos = Position(1, 1, 0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", position=pos)
    assert msg1 == msg2
    
    # Test equality with same values including start/end positions
    start_pos = Position(1, 1, 0)
    end_pos = Position(1, 5, 4)
    msg1 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2
    
    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert msg1 != msg2
    
    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2
    
    # Test inequality with different index
    msg1 = Message(text="Error", index=["field1"])
    msg2 = Message(text="Error", index=["field2"])
    assert msg1 != msg2
    
    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2
    
    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2
    
    # Test inequality with different types
    msg = Message(text="Error")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None
    
    # Test equality with key instead of index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2
    
    # Test equality with position instead of start/end
    pos = Position(1, 1, 0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", start_position=pos, end_position=pos)
    assert msg1 == msg2
    
    # Test hash consistency with equality
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert hash(msg1) == hash(msg2)
    
    # Test that messages with same code and index but different text have same hash
    msg1 = Message(text="Error 1", code="max_length", index=["field"])
    msg2 = Message(text="Error 2", code="max_length", index=["field"])
    assert hash(msg1) == hash(msg2)


# LLM-generated content at query #10
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field"])
    msg2 = Message(text="Error", index=["other"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", start_position=Position(2, 1, 0))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", end_position=Position(1, 2, 0))
    assert msg1 != msg2

    # Test equality with position (start_position and end_position same)
    msg1 = Message(text="Error", position=Position(1, 1, 0))
    msg2 = Message(text="Error", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2

    # Test inequality when one has position and other doesn't
    msg1 = Message(text="Error", position=Position(1, 1, 0))
    msg2 = Message(text="Error")
    assert msg1 != msg2

    # Test equality with key vs index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", index=["users", 3, "username"])
    msg2 = Message(text="Error", index=["users", 3, "username"])
    assert msg1 == msg2


# LLM-generated content at query #11
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 0))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(1, 2, 0))
    assert msg1 != msg2

    # Test equality with position (start_position and end_position same)
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None index
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom", index=[])
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #12
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    assert msg1 == msg2

    # Test equality with index instead of key
    msg1 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    assert msg1 == msg2

    # Test equality with start_position and end_position
    msg1 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(1, 2, 3),
        end_position=Position(1, 5, 6)
    )
    msg2 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(1, 2, 3),
        end_position=Position(1, 5, 6)
    )
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="invalid", key="field")
    msg2 = Message(text="Error 2", code="invalid", key="field")
    assert not (msg1 == msg2)

    # Test inequality with different code
    msg1 = Message(text="Error", code="invalid", key="field")
    msg2 = Message(text="Error", code="required", key="field")
    assert not (msg1 == msg2)

    # Test inequality with different index
    msg1 = Message(text="Error", code="invalid", index=["field1"])
    msg2 = Message(text="Error", code="invalid", index=["field2"])
    assert not (msg1 == msg2)

    # Test inequality with different position
    msg1 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="invalid", key="field", position=Position(2, 3, 4))
    assert not (msg1 == msg2)

    # Test inequality with different start_position
    msg1 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(1, 2, 3),
        end_position=Position(1, 5, 6)
    )
    msg2 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(2, 3, 4),
        end_position=Position(1, 5, 6)
    )
    assert not (msg1 == msg2)

    # Test inequality with different end_position
    msg1 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(1, 2, 3),
        end_position=Position(1, 5, 6)
    )
    msg2 = Message(
        text="Error",
        code="invalid",
        key="field",
        start_position=Position(1, 2, 3),
        end_position=Position(2, 3, 4)
    )
    assert not (msg1 == msg2)

    # Test equality with None positions
    msg1 = Message(text="Error", code="invalid", key="field")
    msg2 = Message(text="Error", code="invalid", key="field")
    assert msg1 == msg2

    # Test inequality with different types
    msg = Message(text="Error", code="invalid", key="field")
    assert not (msg == "not a message")
    assert not (msg == 123)
    assert not (msg == None)

    # Test equality with default code
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="invalid", index=["users", 3, "username"])
    msg2 = Message(text="Error", code="invalid", index=["users", 3, "username"])
    assert msg1 == msg2


# LLM-generated content at query #13
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert msg1 != msg2

    # Test equality with same position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", position=pos)
    msg2 = Message(text="Error", position=pos)
    assert msg1 == msg2

    # Test equality with same start/end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2

    # Test inequality with different start position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2

    # Test inequality with different end position
    pos1 = Position(line_no=1, column_no=5, char_index=4)
    pos2 = Position(line_no=1, column_no=10, char_index=9)
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #14
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test inequality with different text
    msg3 = Message(text="Different", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 != msg3
    
    # Test inequality with different code
    msg4 = Message(text="Error", code="max_length", key="field", position=Position(1, 1, 0))
    assert msg1 != msg4
    
    # Test inequality with different index/key
    msg5 = Message(text="Error", code="custom", key="other_field", position=Position(1, 1, 0))
    assert msg1 != msg5
    
    # Test inequality with different position
    msg6 = Message(text="Error", code="custom", key="field", position=Position(2, 1, 10))
    assert msg1 != msg6
    
    # Test equality with start/end positions instead of single position
    msg7 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 1, 0)
    )
    assert msg1 == msg7
    
    # Test inequality with different start/end positions
    msg8 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 2, 1)
    )
    assert msg1 != msg8
    
    # Test equality with index instead of key
    msg9 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg1 == msg9
    
    # Test inequality with different index
    msg10 = Message(text="Error", code="custom", index=["field", "nested"], position=Position(1, 1, 0))
    assert msg1 != msg10
    
    # Test equality with no position
    msg11 = Message(text="Error", code="custom", key="field")
    msg12 = Message(text="Error", code="custom", key="field")
    assert msg11 == msg12
    
    # Test inequality with None vs position
    assert msg11 != msg1
    
    # Test equality comparison with non-Message object
    assert msg1 != "not a message"
    assert msg1 != 123
    assert msg1 != None
    
    # Test equality with same object
    assert msg1 == msg1
    
    # Test equality with nested index
    msg13 = Message(text="Error", code="custom", index=["users", 3, "username"], position=Position(1, 1, 0))
    msg14 = Message(text="Error", code="custom", index=["users", 3, "username"], position=Position(1, 1, 0))
    assert msg13 == msg14
    
    # Test inequality with different nested index
    msg15 = Message(text="Error", code="custom", index=["users", 4, "username"], position=Position(1, 1, 0))
    assert msg13 != msg15


# LLM-generated content at query #15
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert msg1 != msg2

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality with position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", start_position=None, end_position=None)
    assert msg1 == msg2


# LLM-generated content at query #16
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with same values including positions
    pos = Position(1, 1, 0)
    msg3 = Message(text="Error", code="custom", index=["field"], position=pos)
    msg4 = Message(text="Error", code="custom", index=["field"], position=pos)
    assert msg3 == msg4
    
    # Test inequality with different text
    msg5 = Message(text="Error 1", code="custom", index=["field"])
    msg6 = Message(text="Error 2", code="custom", index=["field"])
    assert not (msg5 == msg6)
    
    # Test inequality with different code
    msg7 = Message(text="Error", code="code1", index=["field"])
    msg8 = Message(text="Error", code="code2", index=["field"])
    assert not (msg7 == msg8)
    
    # Test inequality with different index
    msg9 = Message(text="Error", code="custom", index=["field1"])
    msg10 = Message(text="Error", code="custom", index=["field2"])
    assert not (msg9 == msg10)
    
    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg11 = Message(text="Error", code="custom", index=["field"], start_position=pos1)
    msg12 = Message(text="Error", code="custom", index=["field"], start_position=pos2)
    assert not (msg11 == msg12)
    
    # Test inequality with different end_position
    msg13 = Message(text="Error", code="custom", index=["field"], end_position=pos1)
    msg14 = Message(text="Error", code="custom", index=["field"], end_position=pos2)
    assert not (msg13 == msg14)
    
    # Test equality with None positions
    msg15 = Message(text="Error", code="custom", index=["field"])
    msg16 = Message(text="Error", code="custom", index=["field"])
    assert msg15 == msg16
    
    # Test inequality with different types
    msg17 = Message(text="Error", code="custom", index=["field"])
    assert not (msg17 == "not a Message")
    
    # Test equality with key instead of index
    msg18 = Message(text="Error", code="custom", key="field")
    msg19 = Message(text="Error", code="custom", index=["field"])
    assert msg18 == msg19
    
    # Test equality with position instead of start/end
    msg20 = Message(text="Error", code="custom", index=["field"], position=pos1)
    msg21 = Message(text="Error", code="custom", index=["field"], start_position=pos1, end_position=pos1)
    assert msg20 == msg21
    
    # Test hash equality for equal messages
    msg22 = Message(text="Error", code="custom", index=["field"])
    msg23 = Message(text="Error", code="custom", index=["field"])
    assert hash(msg22) == hash(msg23)


# LLM-generated content at query #17
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 0))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(1, 2, 0))
    assert msg1 != msg2

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error", code="custom")
    assert msg != "not a message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #18
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 10))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(2, 1, 10))
    assert msg1 != msg2

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error", code="custom", index=[])
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg2 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg1 == msg2

    # Test inequality with different position combinations
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom")
    assert msg1 != msg2


# LLM-generated content at query #19
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos2, end_position=pos1)
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", start_position=pos1, end_position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 != msg2

    # Test equality with same position (position parameter)
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", position=pos1)
    assert msg1 == msg2

    # Test equality with same start/end positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with None index vs empty list index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", key="field")
    assert msg1 == msg2

    # Test inequality when one has position and other doesn't
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error")
    assert msg1 != msg2


# LLM-generated content at query #20
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    msg4 = Message(text="Error", code="invalid", index=["field"], position=Position(1, 2, 3))
    assert msg3 == msg4
    
    # Test equality with start/end positions
    msg5 = Message(text="Error", code="invalid", index=["field"], 
                   start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    msg6 = Message(text="Error", code="invalid", index=["field"], 
                   start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    assert msg5 == msg6
    
    # Test inequality with different text
    msg7 = Message(text="Error", code="invalid", key="field")
    msg8 = Message(text="Different", code="invalid", key="field")
    assert msg7 != msg8
    
    # Test inequality with different code
    msg9 = Message(text="Error", code="invalid", key="field")
    msg10 = Message(text="Error", code="required", key="field")
    assert msg9 != msg10
    
    # Test inequality with different index
    msg11 = Message(text="Error", code="invalid", index=["field1"])
    msg12 = Message(text="Error", code="invalid", index=["field2"])
    assert msg11 != msg12
    
    # Test inequality with different positions
    msg13 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg14 = Message(text="Error", code="invalid", key="field", position=Position(2, 3, 4))
    assert msg13 != msg14
    
    # Test inequality with different start/end positions
    msg15 = Message(text="Error", code="invalid", key="field", 
                    start_position=Position(1, 2, 3), end_position=Position(1, 5, 6))
    msg16 = Message(text="Error", code="invalid", key="field", 
                    start_position=Position(1, 2, 3), end_position=Position(1, 6, 7))
    assert msg15 != msg16
    
    # Test inequality with different types
    msg17 = Message(text="Error", code="invalid", key="field")
    assert msg17 != "not a Message"
    assert msg17 != 123
    assert msg17 != None
    
    # Test equality with None positions
    msg18 = Message(text="Error", code="invalid", key="field")
    msg19 = Message(text="Error", code="invalid", key="field")
    assert msg18 == msg19
    
    # Test equality with position vs start/end position (should not be equal)
    msg20 = Message(text="Error", code="invalid", key="field", position=Position(1, 2, 3))
    msg21 = Message(text="Error", code="invalid", key="field", 
                    start_position=Position(1, 2, 3), end_position=Position(1, 2, 3))
    assert msg20 != msg21
    
    # Test equality with same position object
    pos = Position(1, 2, 3)
    msg22 = Message(text="Error", code="invalid", key="field", position=pos)
    msg23 = Message(text="Error", code="invalid", key="field", position=pos)
    assert msg22 == msg23


# LLM-generated content at query #21
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 2, 3))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 2, 3))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", index=["users", 0, "name"])
    msg2 = Message(text="Error", index=["users", 1, "name"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", start_position=Position(1, 1, 1), end_position=Position(1, 5, 5))
    msg2 = Message(text="Error", start_position=Position(2, 1, 6), end_position=Position(2, 5, 10))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", start_position=Position(1, 1, 1), end_position=Position(1, 5, 5))
    msg2 = Message(text="Error", start_position=Position(1, 1, 1), end_position=Position(1, 10, 10))
    assert msg1 != msg2

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", position=Position(1, 2, 3))
    msg2 = Message(text="Error", start_position=Position(1, 2, 3), end_position=Position(1, 2, 3))
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error")
    assert msg1 == msg2

    # Test equality with key vs index
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test hash consistency with equality
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert hash(msg1) == hash(msg2)
    assert msg1 == msg2


# LLM-generated content at query #22
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg4 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg3 == msg4
    
    # Test equality with start/end positions
    msg5 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    msg6 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    assert msg5 == msg6
    
    # Test inequality with different text
    msg7 = Message(text="Error 1", code="custom", key="field")
    msg8 = Message(text="Error 2", code="custom", key="field")
    assert msg7 != msg8
    
    # Test inequality with different code
    msg9 = Message(text="Error", code="max_length", key="field")
    msg10 = Message(text="Error", code="min_length", key="field")
    assert msg9 != msg10
    
    # Test inequality with different index
    msg11 = Message(text="Error", code="custom", index=["field1"])
    msg12 = Message(text="Error", code="custom", index=["field2"])
    assert msg11 != msg12
    
    # Test inequality with different positions
    msg13 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg14 = Message(text="Error", code="custom", position=Position(2, 1, 10))
    assert msg13 != msg14
    
    # Test inequality with different start/end positions
    msg15 = Message(
        text="Error", 
        code="custom", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    msg16 = Message(
        text="Error", 
        code="custom", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 6, 5)
    )
    assert msg15 != msg16
    
    # Test inequality with different types
    msg17 = Message(text="Error", code="custom")
    assert msg17 != "not a Message"
    assert msg17 != 123
    assert msg17 != None
    
    # Test equality with same text but different key/index representation
    msg18 = Message(text="Error", code="custom", key="field")
    msg19 = Message(text="Error", code="custom", index=["field"])
    assert msg18 == msg19
    
    # Test equality with empty index
    msg20 = Message(text="Error", code="custom")
    msg21 = Message(text="Error", code="custom", index=[])
    assert msg20 == msg21
    
    # Test equality with None positions
    msg22 = Message(text="Error", code="custom")
    msg23 = Message(text="Error", code="custom")
    assert msg22 == msg23
    
    # Test equality with position vs start/end position (same position)
    msg24 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg25 = Message(
        text="Error", 
        code="custom", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 1, 0)
    )
    assert msg24 == msg25


# LLM-generated content at query #23
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom", index=["field"])
    msg2 = Message(text="Error 2", code="custom", index=["field"])
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length", index=["field"])
    msg2 = Message(text="Error", code="min_length", index=["field"])
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 10))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 5, 4))
    msg2 = Message(text="Error", code="custom", end_position=Position(1, 10, 9))
    assert msg1 != msg2

    # Test equality with position (start_position and end_position same)
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom", index=[])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg2 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg1 == msg2


# LLM-generated content at query #24
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1")
    msg2 = Message(text="Error 2")
    assert not (msg1 == msg2)

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert not (msg1 == msg2)

    # Test inequality with different index
    msg1 = Message(text="Error", index=["field1"])
    msg2 = Message(text="Error", index=["field2"])
    assert not (msg1 == msg2)

    # Test inequality with different start_position
    pos1 = Position(1, 1, 0)
    pos2 = Position(2, 1, 10)
    msg1 = Message(text="Error", start_position=pos1)
    msg2 = Message(text="Error", start_position=pos2)
    assert not (msg1 == msg2)

    # Test inequality with different end_position
    msg1 = Message(text="Error", end_position=pos1)
    msg2 = Message(text="Error", end_position=pos2)
    assert not (msg1 == msg2)

    # Test equality with same positions
    msg1 = Message(text="Error", start_position=pos1, end_position=pos2)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos2)
    assert msg1 == msg2

    # Test equality with position parameter
    msg1 = Message(text="Error", position=pos1)
    msg2 = Message(text="Error", start_position=pos1, end_position=pos1)
    assert msg1 == msg2

    # Test inequality with different types
    msg = Message(text="Error")
    assert not (msg == "not a Message")
    assert not (msg == 123)
    assert not (msg == None)

    # Test equality with key parameter
    msg1 = Message(text="Error", key="field")
    msg2 = Message(text="Error", index=["field"])
    assert msg1 == msg2

    # Test equality with empty index
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", index=[])
    assert msg1 == msg2

    # Test equality with None positions
    msg1 = Message(text="Error")
    msg2 = Message(text="Error", start_position=None, end_position=None)
    assert msg1 == msg2


# LLM-generated content at query #25
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", key="field")
    assert msg1 == msg2
    
    # Test equality with same values using index instead of key
    msg3 = Message(text="Error", code="custom", index=["field"])
    msg4 = Message(text="Error", code="custom", index=["field"])
    assert msg3 == msg4
    
    # Test equality with position
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg5 = Message(text="Error", code="custom", position=pos)
    msg6 = Message(text="Error", code="custom", position=pos)
    assert msg5 == msg6
    
    # Test equality with start/end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg7 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    msg8 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    assert msg7 == msg8
    
    # Test inequality with different text
    msg9 = Message(text="Error 1", code="custom")
    msg10 = Message(text="Error 2", code="custom")
    assert msg9 != msg10
    
    # Test inequality with different code
    msg11 = Message(text="Error", code="code1")
    msg12 = Message(text="Error", code="code2")
    assert msg11 != msg12
    
    # Test inequality with different index
    msg13 = Message(text="Error", code="custom", index=["field1"])
    msg14 = Message(text="Error", code="custom", index=["field2"])
    assert msg13 != msg14
    
    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg15 = Message(text="Error", code="custom", position=pos1)
    msg16 = Message(text="Error", code="custom", position=pos2)
    assert msg15 != msg16
    
    # Test inequality with different start_position
    start_pos1 = Position(line_no=1, column_no=1, char_index=0)
    start_pos2 = Position(line_no=2, column_no=1, char_index=10)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg17 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos)
    msg18 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos)
    assert msg17 != msg18
    
    # Test inequality with different end_position
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos1 = Position(line_no=1, column_no=5, char_index=4)
    end_pos2 = Position(line_no=1, column_no=10, char_index=9)
    msg19 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos1)
    msg20 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos2)
    assert msg19 != msg20
    
    # Test equality with None positions
    msg21 = Message(text="Error", code="custom")
    msg22 = Message(text="Error", code="custom")
    assert msg21 == msg22
    
    # Test inequality with different types
    msg23 = Message(text="Error", code="custom")
    assert msg23 != "not a Message"
    assert msg23 != 123
    assert msg23 != None
    
    # Test equality with complex index
    msg24 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg25 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg24 == msg25
    
    # Test inequality with different length index
    msg26 = Message(text="Error", code="custom", index=["field"])
    msg27 = Message(text="Error", code="custom", index=["field", "subfield"])
    assert msg26 != msg27


# LLM-generated content at query #26
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg4 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg3 == msg4
    
    # Test inequality with different text
    msg5 = Message(text="Error 1", code="custom", key="field")
    msg6 = Message(text="Error 2", code="custom", key="field")
    assert msg5 != msg6
    
    # Test inequality with different code
    msg7 = Message(text="Error", code="max_length", key="field")
    msg8 = Message(text="Error", code="min_length", key="field")
    assert msg7 != msg8
    
    # Test inequality with different index
    msg9 = Message(text="Error", code="custom", index=["field1"])
    msg10 = Message(text="Error", code="custom", index=["field2"])
    assert msg9 != msg10
    
    # Test inequality with different start_position
    msg11 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    msg12 = Message(text="Error", code="custom", start_position=Position(2, 1, 10), end_position=Position(2, 5, 14))
    assert msg11 != msg12
    
    # Test inequality with different end_position
    msg13 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    msg14 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 10, 9))
    assert msg13 != msg14
    
    # Test equality with position (same start and end)
    msg15 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg16 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg15 == msg16
    
    # Test inequality with different object type
    msg17 = Message(text="Error", code="custom")
    assert msg17 != "not a Message"
    assert msg17 != 123
    assert msg17 != None
    
    # Test equality with empty index
    msg18 = Message(text="Error", code="custom")
    msg19 = Message(text="Error", code="custom", index=[])
    assert msg18 == msg19
    
    # Test equality with None positions
    msg20 = Message(text="Error", code="custom")
    msg21 = Message(text="Error", code="custom")
    assert msg20 == msg21
    
    # Test equality with complex index
    msg22 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg23 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg22 == msg23


# LLM-generated content at query #27
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    assert not (msg1 != msg2)

    # Test equality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test equality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2

    # Test equality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test equality with different positions
    pos1 = Position(line_no=1, column_no=1, char_index=0)
    pos2 = Position(line_no=2, column_no=1, char_index=10)
    msg1 = Message(text="Error", code="custom", position=pos1)
    msg2 = Message(text="Error", code="custom", position=pos2)
    assert msg1 != msg2

    # Test equality with start/end positions
    start_pos = Position(line_no=1, column_no=1, char_index=0)
    end_pos = Position(line_no=1, column_no=5, char_index=4)
    msg1 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    msg2 = Message(text="Error", code="custom", start_position=start_pos, end_position=end_pos)
    assert msg1 == msg2

    # Test equality with different start/end positions
    start_pos1 = Position(line_no=1, column_no=1, char_index=0)
    end_pos1 = Position(line_no=1, column_no=5, char_index=4)
    start_pos2 = Position(line_no=2, column_no=1, char_index=10)
    end_pos2 = Position(line_no=2, column_no=5, char_index=14)
    msg1 = Message(text="Error", code="custom", start_position=start_pos1, end_position=end_pos1)
    msg2 = Message(text="Error", code="custom", start_position=start_pos2, end_position=end_pos2)
    assert msg1 != msg2

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with mixed position types
    pos = Position(line_no=1, column_no=1, char_index=0)
    msg1 = Message(text="Error", code="custom", position=pos)
    msg2 = Message(text="Error", code="custom", start_position=pos, end_position=pos)
    assert msg1 == msg2

    # Test equality with different object type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"
    assert msg != 123
    assert msg != None

    # Test equality with same key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with different key/index
    msg1 = Message(text="Error", code="custom", key="field1")
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2

    # Test equality with empty index
    msg1 = Message(text="Error", code="custom", index=[])
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2

    # Test equality with different complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 1, "name"])
    assert msg1 != msg2


# LLM-generated content at query #28
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1")
    msg2 = Message(text="Error", code="code2")
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field"])
    msg2 = Message(text="Error", code="custom", index=["other"])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 0))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(1, 2, 0))
    assert msg1 != msg2

    # Test equality with position instead of start/end
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2


# LLM-generated content at query #29
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom")
    msg2 = Message(text="Error 2", code="custom")
    assert msg1 != msg2
    
    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length")
    msg2 = Message(text="Error", code="min_length")
    assert msg1 != msg2
    
    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=["field1"])
    msg2 = Message(text="Error", code="custom", index=["field2"])
    assert msg1 != msg2
    
    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(2, 1, 10))
    assert msg1 != msg2
    
    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", end_position=Position(2, 1, 10))
    assert msg1 != msg2
    
    # Test equality with position (start_position and end_position same)
    msg1 = Message(text="Error", code="custom", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test inequality with different object type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"
    
    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2
    
    # Test equality with empty index
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom", index=[])
    assert msg1 == msg2
    
    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom", start_position=None, end_position=None)
    assert msg1 == msg2


# LLM-generated content at query #30
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom", index=[0])
    msg2 = Message(text="Error 2", code="custom", index=[0])
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="max_length", index=[0])
    msg2 = Message(text="Error", code="min_length", index=[0])
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[1])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 2, 1))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", index=[0], end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], end_position=Position(1, 2, 1))
    assert msg1 != msg2

    # Test equality with position (start and end same)
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error", code="custom", index=[0])
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[0])
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="username")
    msg2 = Message(text="Error", code="custom", index=["username"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg2 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg1 == msg2

    # Test inequality with mixed position specifications
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 2, 1))
    assert msg1 != msg2


# LLM-generated content at query #31
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    msg4 = Message(text="Error", code="custom", index=["field"], position=Position(1, 1, 0))
    assert msg3 == msg4
    
    # Test equality with start_position and end_position
    msg5 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    msg6 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    assert msg5 == msg6
    
    # Test inequality with different text
    msg7 = Message(text="Error 1", code="custom", key="field")
    msg8 = Message(text="Error 2", code="custom", key="field")
    assert msg7 != msg8
    
    # Test inequality with different code
    msg9 = Message(text="Error", code="max_length", key="field")
    msg10 = Message(text="Error", code="min_length", key="field")
    assert msg9 != msg10
    
    # Test inequality with different index
    msg11 = Message(text="Error", code="custom", index=["field1"])
    msg12 = Message(text="Error", code="custom", index=["field2"])
    assert msg11 != msg12
    
    # Test inequality with different position
    msg13 = Message(text="Error", code="custom", key="field", position=Position(1, 1, 0))
    msg14 = Message(text="Error", code="custom", key="field", position=Position(2, 1, 10))
    assert msg13 != msg14
    
    # Test inequality with different start_position
    msg15 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    msg16 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(2, 1, 10),
        end_position=Position(1, 5, 4)
    )
    assert msg15 != msg16
    
    # Test inequality with different end_position
    msg17 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 5, 4)
    )
    msg18 = Message(
        text="Error", 
        code="custom", 
        key="field", 
        start_position=Position(1, 1, 0),
        end_position=Position(1, 10, 9)
    )
    assert msg17 != msg18
    
    # Test equality with None positions
    msg19 = Message(text="Error", code="custom", key="field")
    msg20 = Message(text="Error", code="custom", key="field")
    assert msg19 == msg20
    
    # Test inequality with different types
    msg21 = Message(text="Error", code="custom", key="field")
    assert msg21 != "not a Message"
    assert msg21 != 123
    assert msg21 != None
    
    # Test equality with empty index
    msg22 = Message(text="Error", code="custom")
    msg23 = Message(text="Error", code="custom")
    assert msg22 == msg23
    
    # Test equality with complex index
    msg24 = Message(text="Error", code="custom", index=["users", 3, "username"])
    msg25 = Message(text="Error", code="custom", index=["users", 3, "username"])
    assert msg24 == msg25


# LLM-generated content at query #32
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different text
    msg1 = Message(text="Error 1", code="custom", index=[0])
    msg2 = Message(text="Error 2", code="custom", index=[0])
    assert msg1 != msg2

    # Test inequality with different code
    msg1 = Message(text="Error", code="code1", index=[0])
    msg2 = Message(text="Error", code="code2", index=[0])
    assert msg1 != msg2

    # Test inequality with different index
    msg1 = Message(text="Error", code="custom", index=[0])
    msg2 = Message(text="Error", code="custom", index=[1])
    assert msg1 != msg2

    # Test inequality with different start_position
    msg1 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(2, 1, 10))
    assert msg1 != msg2

    # Test inequality with different end_position
    msg1 = Message(text="Error", code="custom", index=[0], end_position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], end_position=Position(2, 1, 10))
    assert msg1 != msg2

    # Test equality with position (start_position and end_position same)
    msg1 = Message(text="Error", code="custom", index=[0], position=Position(1, 1, 0))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 1, 0))
    assert msg1 == msg2

    # Test inequality with different object type
    msg = Message(text="Error", code="custom")
    assert msg != "not a Message"

    # Test equality with None positions
    msg1 = Message(text="Error", code="custom")
    msg2 = Message(text="Error", code="custom")
    assert msg1 == msg2

    # Test equality with key instead of index
    msg1 = Message(text="Error", code="custom", key="field")
    msg2 = Message(text="Error", code="custom", index=["field"])
    assert msg1 == msg2

    # Test equality with complex index
    msg1 = Message(text="Error", code="custom", index=["users", 0, "name"])
    msg2 = Message(text="Error", code="custom", index=["users", 0, "name"])
    assert msg1 == msg2

    # Test inequality with different position but same other attributes
    msg1 = Message(text="Error", code="custom", index=[0], start_position=Position(1, 1, 0), end_position=Position(1, 5, 4))
    msg2 = Message(text="Error", code="custom", index=[0], start_position=Position(2, 1, 10), end_position=Position(2, 5, 14))
    assert msg1 != msg2


# LLM-generated content at query #33
#--------------------------

```python
def test_Message___eq__():
    # Test equality with same values
    msg1 = Message(text="Error", code="max_length", key="username")
    msg2 = Message(text="Error", code="max_length", key="username")
    assert msg1 == msg2
    
    # Test equality with index instead of key
    msg3 = Message(text="Error", code="max_length", index=["username"])
    msg4 = Message(text="Error", code="max_length", index=["username"])
    assert msg3 == msg4
    
    # Test equality with position
    pos = Position(line_no=1, column_no=5, char_index=10)
    msg5 = Message(text="Error", code="max_length", position=pos)
    msg6 = Message(text="Error", code="max_length", position=pos)
    assert msg5 == msg6
    
    # Test equality with start/end position
    start_pos = Position(line_no=1, column_no=5, char_index=10)
    end_pos = Position(line_no=1, column_no=10, char_index=15)
    msg7 = Message(text="Error", code="max_length", start_position=start_pos, end_position=end_pos)
    msg8 = Message(text="Error", code="max_length", start_position=start_pos, end_position=end_pos)
    assert msg7 == msg8
    
    # Test inequality with different text
    msg9 = Message(text="Error 1", code="max_length")
    msg10 = Message(text="Error 2", code="max_length")
    assert msg9 != msg10
    
    # Test inequality with different code
    msg11 = Message(text="Error", code="max_length")
    msg12 = Message(text="Error", code="min_length")
    assert msg11 != msg12
    
    # Test inequality with different index
    msg13 = Message(text="Error", code="max_length", index=["username"])
    msg14 = Message(text="Error", code="max_length", index=["email"])
    assert msg13 != msg14
    
    # Test inequality with different position
    pos1 = Position(line_no=1, column_no=5, char_index=10)
    pos2 = Position(line_no=2, column_no=5, char_index=20)
    msg15 = Message(text="Error", code="max_length", position=pos1)
    msg16 = Message(text="Error", code="max_length", position=pos2)
    assert msg15 != msg16
    
    # Test inequality with different start/end position
    start_pos1 = Position(line_no=1, column_no=5, char_index=10)
    end_pos1 = Position(line_no=1, column_no=10, char_index=15)
    start_pos2 = Position(line_no=2, column_no=5, char_index=20)
    end_pos2 = Position(line_no=2, column_no=10, char_index=25)
    msg17 = Message(text="Error", code="max_length", start_position=start_pos1, end_position=end_pos1)
    msg18 = Message(text="Error", code="max_length", start_position=start_pos2, end_position=end_pos2)
    assert msg17 != msg18
    
    # Test inequality with None position vs actual position
    msg19 = Message(text="Error", code="max_length")
    msg20 = Message(text="Error", code="max_length", position=pos1)
    assert msg19 != msg20
    
    # Test equality with None code (should default to "custom")
    msg21 = Message(text="Error")
    msg22 = Message(text="Error", code="custom")
    assert msg21 == msg22
    
    # Test inequality with different object type
    msg23 = Message(text="Error", code="max_length")
    assert msg23 != "not a Message"
    assert msg23 != 123
    assert msg23 != None
    
    # Test equality with empty index
    msg24 = Message(text="Error", code="max_length", index=[])
    msg25 = Message(text="Error", code="max_length")
    assert msg24 == msg25
    
    # Test equality with complex index
    msg26 = Message(text="Error", code="max_length", index=["users", 3, "username"])
    msg27 = Message(text="Error", code="max_length", index=["users", 3, "username"])
    assert msg26 == msg27


