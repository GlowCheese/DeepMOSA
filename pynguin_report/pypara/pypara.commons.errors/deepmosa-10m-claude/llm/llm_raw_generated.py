####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert str(error) == ""


def test_programming_error_is_exception():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_can_be_raised():
    try:
        raise ProgrammingError("Test error")
    except ProgrammingError as e:
        assert str(e) == "Test error"


def test_programming_error_passert_with_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)


def test_programming_error_passert_with_false_condition_and_default_message():
    try:
        ProgrammingError.passert(False)
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_false_condition_and_custom_message():
    try:
        ProgrammingError.passert(False, "Custom assertion message")
    except ProgrammingError as e:
        assert str(e) == "Custom assertion message"


def test_programming_error_passert_with_true_condition_and_message():
    ProgrammingError.passert(True, "This should not raise")


# LLM-generated content at query #2
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception_subclass():
    error = ProgrammingError("test")
    assert isinstance(error, Exception)


def test_programming_error_passert_with_true_condition():
    ProgrammingError.passert(True)


def test_programming_error_passert_with_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_with_true_condition_and_message():
    ProgrammingError.passert(True, "This should not raise")


# LLM-generated content at query #3
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    custom_message = "Custom error message"
    try:
        ProgrammingError.passert(False, custom_message)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == custom_message


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert isinstance(e, ProgrammingError)


# LLM-generated content at query #4
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert isinstance(e, ProgrammingError)


# LLM-generated content at query #5
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception_subclass():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""


# LLM-generated content at query #6
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""
    assert isinstance(error, Exception)


# LLM-generated content at query #7
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_condition_true():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_condition_false_no_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_condition_false_with_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_complex_condition():
    ProgrammingError.passert(5 > 3, "Five should be greater than three")
    try:
        ProgrammingError.passert(5 < 3, "Five should be less than three")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Five should be less than three"


# LLM-generated content at query #8
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_condition_true():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_condition_false():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_condition_false_with_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_condition_false_with_none_message():
    try:
        ProgrammingError.passert(False, None)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #9
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_with_expression():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #10
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""


# LLM-generated content at query #11
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)


def test_programming_error_passert_true_condition_with_message():
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_with_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_numeric_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #12
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_empty_string_message():
    try:
        ProgrammingError.passert(False, "")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #13
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    custom_message = "Custom error message"
    try:
        ProgrammingError.passert(False, custom_message)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == custom_message


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert isinstance(e, ProgrammingError)


# LLM-generated content at query #14
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert isinstance(e, ProgrammingError)


# LLM-generated content at query #15
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_with_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_with_empty_message():
    try:
        ProgrammingError.passert(False, "")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #16
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_can_be_raised():
    try:
        raise ProgrammingError("Test error")
    except ProgrammingError as e:
        assert str(e) == "Test error"


def test_programming_error_passert_with_true_condition():
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_with_false_condition_and_message():
    try:
        ProgrammingError.passert(False, "Custom message")
    except ProgrammingError as e:
        assert str(e) == "Custom message"


def test_programming_error_passert_with_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_true_condition_and_message():
    ProgrammingError.passert(True, "Custom message")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_with_message():
    custom_message = "Custom error message"
    try:
        ProgrammingError.passert(False, custom_message)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == custom_message


def test_programming_error_passert_false_condition_with_empty_message():
    try:
        ProgrammingError.passert(False, "")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #2
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_false_condition_with_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_without_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_expression():
    ProgrammingError.passert(5 > 3, "Five should be greater than three")


def test_programming_error_passert_false_with_expression():
    try:
        ProgrammingError.passert(5 < 3, "Five is not less than three")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Five is not less than three"


# LLM-generated content at query #3
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_with_true_condition():
    ProgrammingError.passert(True)


def test_programming_error_passert_with_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_false_condition_and_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_with_true_condition_and_message():
    ProgrammingError.passert(True, "This message should not appear")


def test_programming_error_is_exception_subclass():
    error = ProgrammingError("Test")
    assert isinstance(error, Exception)


# LLM-generated content at query #4
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception_subclass():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""


# LLM-generated content at query #5
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_empty_message():
    try:
        ProgrammingError.passert(1 == 0, "")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #6
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_args():
    error = ProgrammingError("Error", "Extra", "Args")
    assert isinstance(error, Exception)


# LLM-generated content at query #7
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_with_true_condition():
    ProgrammingError.passert(True)


def test_programming_error_passert_with_true_condition_and_message():
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_with_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_false_condition_and_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_with_false_condition_and_empty_message():
    try:
        ProgrammingError.passert(False, "")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #8
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""


def test_programming_error_can_be_raised():
    try:
        raise ProgrammingError("Test exception")
    except ProgrammingError as e:
        assert str(e) == "Test exception"


# LLM-generated content at query #9
#--------------------------

```python
def test_programming_error_constructor_with_message():
    error = ProgrammingError("Custom error message")
    assert str(error) == "Custom error message"


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception():
    error = ProgrammingError("Test error")
    assert isinstance(error, Exception)


def test_programming_error_constructor_with_empty_string():
    error = ProgrammingError("")
    assert str(error) == ""


def test_programming_error_constructor_inheritance():
    error = ProgrammingError("Test")
    assert isinstance(error, BaseException)


# LLM-generated content at query #10
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_args():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_with_message():
    custom_message = "Custom error message"
    try:
        ProgrammingError.passert(False, custom_message)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == custom_message


def test_programming_error_passert_false_condition_no_message():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #11
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_without_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_inheritance():
    error = ProgrammingError("Test")
    assert isinstance(error, Exception)
    assert isinstance(error, ProgrammingError)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_with_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_with_empty_message():
    try:
        ProgrammingError.passert(False, "")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


# LLM-generated content at query #12
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_with_expression():
    try:
        ProgrammingError.passert(1 == 0, "Numbers don't match")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Numbers don't match"


# LLM-generated content at query #13
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert str(error) == ""
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True, "This should not raise")


def test_programming_error_passert_false_condition_with_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_without_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_with_expression():
    ProgrammingError.passert(1 + 1 == 2, "Math is broken")
    try:
        ProgrammingError.passert(1 + 1 == 3, "Math is broken")
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Math is broken"


# LLM-generated content at query #14
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_no_message():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Custom message")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    custom_message = "Custom error message"
    try:
        ProgrammingError.passert(False, custom_message)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert str(e) == custom_message


def test_programming_error_passert_false_condition():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Expected ProgrammingError to be raised"
    except ProgrammingError as e:
        assert isinstance(e, ProgrammingError)


# LLM-generated content at query #15
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_programming_error_constructor_default():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True)
    ProgrammingError.passert(1 == 1)
    ProgrammingError.passert(True, "Should not raise")


def test_programming_error_passert_false_condition_default_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_passert_false_condition_custom_message():
    try:
        ProgrammingError.passert(False, "Custom error message")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Custom error message"


def test_programming_error_passert_false_condition_with_expression():
    try:
        ProgrammingError.passert(1 == 0)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


def test_programming_error_inheritance():
    error = ProgrammingError("Test")
    assert isinstance(error, Exception)


# LLM-generated content at query #16
#--------------------------

```python
def test_programming_error_constructor():
    error = ProgrammingError("test message")
    assert str(error) == "test message"


def test_programming_error_constructor_no_args():
    error = ProgrammingError()
    assert isinstance(error, Exception)


def test_programming_error_is_exception():
    error = ProgrammingError("test")
    assert isinstance(error, Exception)


def test_programming_error_passert_true_condition():
    ProgrammingError.passert(True, "should not raise")


def test_programming_error_passert_false_condition_with_message():
    try:
        ProgrammingError.passert(False, "custom error message")
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "custom error message"


def test_programming_error_passert_false_condition_without_message():
    try:
        ProgrammingError.passert(False)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Broken coherence. Check your code against domain logic to fix it."


