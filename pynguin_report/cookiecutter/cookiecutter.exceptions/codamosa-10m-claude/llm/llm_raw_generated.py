####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UnknownExtension():
    """Test UnknownExtension exception can be instantiated and raised."""
    # Test basic instantiation
    exc = UnknownExtension()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Failed to import extension"
    exc = UnknownExtension(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(UnknownExtension):
        raise UnknownExtension("Test extension error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownExtension("Test extension error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise UnknownExtension("Test extension error")
    
    # Test with multiple arguments
    exc = UnknownExtension("arg1", "arg2")
    assert exc.args == ("arg1", "arg2")


# LLM-generated content at query #2
#--------------------------

```python
def test_InvalidConfiguration():
    """Test InvalidConfiguration exception can be instantiated and raised."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Configuration file is not valid YAML"
    exc = InvalidConfiguration(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration("Invalid config")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise InvalidConfiguration("Invalid config")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise InvalidConfiguration("Invalid config")


# LLM-generated content at query #3
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test RepositoryCloneFailed exception initialization and inheritance."""
    # Test basic instantiation with a message
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, RepositoryCloneFailed)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == message
    assert exception.args == (message,)
    
    # Test instantiation without arguments
    exception_no_args = RepositoryCloneFailed()
    assert isinstance(exception_no_args, RepositoryCloneFailed)
    assert str(exception_no_args) == ""
    
    # Test instantiation with multiple arguments
    exception_multi = RepositoryCloneFailed("Error 1", "Error 2")
    assert exception_multi.args == ("Error 1", "Error 2")
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryCloneFailed):
        raise RepositoryCloneFailed("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryCloneFailed("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryCloneFailed("Test error")


# LLM-generated content at query #4
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test RepositoryCloneFailed exception can be instantiated and raised."""
    # Test basic instantiation
    exc = RepositoryCloneFailed("Test message")
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == "Test message"
    
    # Test instantiation without message
    exc_no_msg = RepositoryCloneFailed()
    assert isinstance(exc_no_msg, RepositoryCloneFailed)
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryCloneFailed):
        raise RepositoryCloneFailed("Clone failed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryCloneFailed("Clone failed")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryCloneFailed("Clone failed")
    
    # Test with different message types
    exc_with_args = RepositoryCloneFailed("Error", "Extra", "Args")
    assert isinstance(exc_with_args, RepositoryCloneFailed)


# LLM-generated content at query #5
#--------------------------

```python
def test_ContextDecodingException():
    """Test ContextDecodingException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = ContextDecodingException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Failed to decode context JSON"
    exc = ContextDecodingException(message)
    assert str(exc) == message
    
    # Test raising and catching the exception
    with pytest.raises(ContextDecodingException):
        raise ContextDecodingException("Context decoding failed")
    
    # Test exception inheritance chain
    with pytest.raises(CookiecutterException):
        raise ContextDecodingException("Test error")
    
    # Test with multiple arguments
    exc = ContextDecodingException("Error", "Additional info")
    assert isinstance(exc, ContextDecodingException)


# LLM-generated content at query #6
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test basic instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with message
    message = "Repository not found at /path/to/repo"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    
    # Test raising the exception
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("test repo not found")
    
    # Test raising with message and checking message
    with pytest.raises(RepositoryNotFound) as exc_info:
        raise RepositoryNotFound("Custom repository error")
    assert str(exc_info.value) == "Custom repository error"
    
    # Test exception inheritance chain
    exception = RepositoryNotFound("test")
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #7
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test ConfigDoesNotExistException can be instantiated and raised."""
    # Test instantiation with message
    exc = ConfigDoesNotExistException("Config file not found at /path/to/config")
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == "Config file not found at /path/to/config"
    
    # Test instantiation without message
    exc_no_msg = ConfigDoesNotExistException()
    assert isinstance(exc_no_msg, CookiecutterException)
    
    # Test raising the exception
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        raise ConfigDoesNotExistException("Test error message")
    assert "Test error message" in str(exc_info.value)
    
    # Test that it's a subclass of CookiecutterException
    assert issubclass(ConfigDoesNotExistException, CookiecutterException)
    assert issubclass(ConfigDoesNotExistException, Exception)


# LLM-generated content at query #8
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException can be instantiated and raised."""
    # Test basic instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with message
    message = "Output directory already exists"
    exception_with_msg = OutputDirExistsException(message)
    assert str(exception_with_msg) == message
    
    # Test raising the exception
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Test output directory exists")
    
    # Test exception inheritance chain
    with pytest.raises(CookiecutterException):
        raise OutputDirExistsException("Test message")
    
    # Test with args
    exception_with_args = OutputDirExistsException("arg1", "arg2")
    assert exception_with_args.args == ("arg1", "arg2")


# LLM-generated content at query #9
#--------------------------

```python
def test_InvalidModeException():
    """Test InvalidModeException can be instantiated and raised."""
    # Test instantiation with a message
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test raising the exception
    with pytest.raises(InvalidModeException):
        raise InvalidModeException("no_input and replay cannot both be True")
    
    # Test with no message
    exception_no_msg = InvalidModeException()
    assert isinstance(exception_no_msg, CookiecutterException)


# LLM-generated content at query #10
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test NonTemplatedInputDirException can be instantiated and raised."""
    exception = NonTemplatedInputDirException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"
    
    with pytest.raises(NonTemplatedInputDirException):
        raise NonTemplatedInputDirException("Input directory is not templated")
    
    with pytest.raises(CookiecutterException):
        raise NonTemplatedInputDirException("Base class catch test")


# LLM-generated content at query #11
#--------------------------

```python
def test_CookiecutterException():
    """Test CookiecutterException can be instantiated and raised."""
    # Test basic instantiation
    exc = CookiecutterException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, Exception)
    
    # Test raising and catching
    with pytest.raises(CookiecutterException):
        raise CookiecutterException("Test error")
    
    # Test with no message
    exc_no_msg = CookiecutterException()
    assert isinstance(exc_no_msg, CookiecutterException)
    
    # Test inheritance
    assert issubclass(CookiecutterException, Exception)


# LLM-generated content at query #12
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test __str__ method of UndefinedVariableInTemplate exception."""
    from jinja2 import TemplateError
    
    # Create a mock TemplateError
    template_error = TemplateError()
    template_error.message = "Variable 'foo' is undefined"
    
    # Create test context
    test_context = {"name": "test_project", "version": "1.0"}
    
    # Create the exception
    message = "Error rendering template"
    exc = UndefinedVariableInTemplate(message, template_error, test_context)
    
    # Test __str__ output
    result = str(exc)
    
    # Verify the result contains all expected parts
    assert "Error rendering template" in result
    assert "Variable 'foo' is undefined" in result
    assert "test_project" in result
    assert "1.0" in result
    assert "Error message:" in result
    assert "Context:" in result


def test_UndefinedVariableInTemplate___str___empty_context():
    """Test __str__ method with empty context."""
    from jinja2 import TemplateError
    
    template_error = TemplateError()
    template_error.message = "No variable found"
    
    message = "Template error occurred"
    exc = UndefinedVariableInTemplate(message, template_error, {})
    
    result = str(exc)
    
    assert "Template error occurred" in result
    assert "No variable found" in result
    assert "Context: {}" in result


def test_UndefinedVariableInTemplate___str___complex_context():
    """Test __str__ method with complex nested context."""
    from jinja2 import TemplateError
    
    template_error = TemplateError()
    template_error.message = "Complex variable error"
    
    complex_context = {
        "project": {
            "name": "my_project",
            "nested": {"value": 42}
        },
        "list_var": [1, 2, 3]
    }
    
    message = "Complex template rendering failed"
    exc = UndefinedVariableInTemplate(message, template_error, complex_context)
    
    result = str(exc)
    
    assert "Complex template rendering failed" in result
    assert "Complex variable error" in result
    assert "my_project" in result
    assert "42" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from jinja2 import TemplateError
    
    # Create a mock TemplateError
    template_error = TemplateError()
    template_error.message = "undefined variable 'foo'"
    
    # Test data
    message = "Variable 'foo' is undefined"
    context = {"bar": "value", "baz": 123}
    
    # Initialize the exception
    exception = UndefinedVariableInTemplate(message, template_error, context)
    
    # Assert attributes are set correctly
    assert exception.message == message
    assert exception.error == template_error
    assert exception.context == context
    
    # Assert string representation
    expected_str = (
        f"{message}. "
        f"Error message: {template_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
    
    # Test with different context
    context2 = {"key1": "val1", "key2": [1, 2, 3]}
    exception2 = UndefinedVariableInTemplate("Another error", template_error, context2)
    
    assert exception2.message == "Another error"
    assert exception2.context == context2
    assert "Another error" in str(exception2)
    assert str(context2) in str(exception2)
    
    # Test with empty context
    exception3 = UndefinedVariableInTemplate("Empty context error", template_error, {})
    assert exception3.context == {}
    assert "{}" in str(exception3)


# LLM-generated content at query #14
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test ConfigDoesNotExistException constructor and inheritance."""
    # Test that exception can be instantiated with a message
    message = "Config file not found at /path/to/config.yaml"
    exception = ConfigDoesNotExistException(message)
    
    # Verify the exception message
    assert str(exception) == message
    
    # Verify it's an instance of CookiecutterException
    assert isinstance(exception, CookiecutterException)
    
    # Verify it's an instance of Exception
    assert isinstance(exception, Exception)
    
    # Test that exception can be raised and caught
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        raise ConfigDoesNotExistException("Test error message")
    
    assert str(exc_info.value) == "Test error message"
    
    # Test that exception can be instantiated without arguments
    exception_no_args = ConfigDoesNotExistException()
    assert isinstance(exception_no_args, ConfigDoesNotExistException)


# LLM-generated content at query #15
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from jinja2 import TemplateError
    
    # Create a mock TemplateError
    template_error = TemplateError("Variable 'foo' is undefined")
    template_error.message = "Variable 'foo' is undefined"
    
    # Create test context
    context = {"var1": "value1", "var2": "value2"}
    
    # Test initialization
    message = "Template variable error"
    exc = UndefinedVariableInTemplate(message, template_error, context)
    
    # Verify attributes are set correctly
    assert exc.message == message
    assert exc.error == template_error
    assert exc.context == context
    
    # Test string representation
    str_repr = str(exc)
    assert message in str_repr
    assert "Variable 'foo' is undefined" in str_repr
    assert "var1" in str_repr
    assert "value1" in str_repr
    
    # Test that it's a CookiecutterException
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #16
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test instantiation with no message
    exc = RepositoryNotFound()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Repository not found at specified path"
    exc = RepositoryNotFound(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Test repository not found")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryNotFound("Test repository not found")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryNotFound("Test repository not found")
    
    # Test with multiple arguments
    exc = RepositoryNotFound("Repository", "not", "found")
    assert isinstance(exc, RepositoryNotFound)


# LLM-generated content at query #17
#--------------------------

```python
def test_UnknownRepoType():
    """Test UnknownRepoType exception can be instantiated and raised."""
    # Test instantiation with no message
    exc = UnknownRepoType()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Unable to determine repository type"
    exc = UnknownRepoType(message)
    assert str(exc) == message
    assert exc.args == (message,)

    # Test raising the exception
    with pytest.raises(UnknownRepoType):
        raise UnknownRepoType("Repository type is unknown")

    # Test catching as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownRepoType("Repository type is unknown")

    # Test catching as Exception
    with pytest.raises(Exception):
        raise UnknownRepoType("Repository type is unknown")


# LLM-generated content at query #18
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test string representation of UndefinedVariableInTemplate exception."""
    # Create a mock TemplateError
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    # Test data
    message = "Variable 'project_name' is undefined"
    error = MockTemplateError("jinja2.exceptions.UndefinedError: 'project_name' is undefined")
    context = {"author": "John Doe", "version": "1.0"}

    # Create exception instance
    exception = UndefinedVariableInTemplate(message, error, context)

    # Test __str__ output
    result = str(exception)

    # Verify the result contains all expected parts
    assert message in result
    assert error.message in result
    assert "Context:" in result
    assert "author" in result
    assert "John Doe" in result
    assert "version" in result
    assert "1.0" in result
    assert "Error message:" in result

    # Verify the exact format
    expected = (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )
    assert result == expected


def test_UndefinedVariableInTemplate___str___empty_context():
    """Test string representation with empty context."""
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    message = "Variable is undefined"
    error = MockTemplateError("undefined error")
    context = {}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert message in result
    assert error.message in result
    assert "Context: {}" in result


def test_UndefinedVariableInTemplate___str___special_characters():
    """Test string representation with special characters in message."""
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    message = "Variable '{{ undefined_var }}' is not defined"
    error = MockTemplateError("UndefinedError: '{{ undefined_var }}'")
    context = {"key": "value with 'quotes' and \"double quotes\""}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert message in result
    assert error.message in result
    assert "Context:" in result


# LLM-generated content at query #19
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test the __str__ method of UndefinedVariableInTemplate."""
    # Create a mock TemplateError object
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    error = MockTemplateError("variable 'foo' is undefined")
    context = {"key1": "value1", "key2": "value2"}
    message = "Template variable error"
    
    exception = UndefinedVariableInTemplate(message, error, context)
    
    result = str(exception)
    
    assert "Template variable error" in result
    assert "variable 'foo' is undefined" in result
    assert "key1" in result
    assert "value1" in result
    assert "key2" in result
    assert "value2" in result
    assert "Error message:" in result
    assert "Context:" in result


def test_UndefinedVariableInTemplate___str___empty_context():
    """Test __str__ method with empty context."""
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    error = MockTemplateError("undefined variable")
    context = {}
    message = "Error"
    
    exception = UndefinedVariableInTemplate(message, error, context)
    
    result = str(exception)
    
    assert "Error" in result
    assert "undefined variable" in result
    assert "Context: {}" in result


def test_UndefinedVariableInTemplate___str___special_characters():
    """Test __str__ method with special characters in message and context."""
    class MockTemplateError:
        def __init__(self, message: str):
            self.message = message

    error = MockTemplateError("error with 'quotes' and \"double quotes\"")
    context = {"special": "value with 'special' chars"}
    message = "Message with special chars: !@#$%"
    
    exception = UndefinedVariableInTemplate(message, error, context)
    
    result = str(exception)
    
    assert "Message with special chars: !@#$%" in result
    assert "error with 'quotes' and \"double quotes\"" in result
    assert "special" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_EmptyDirNameException():
    """Test EmptyDirNameException can be instantiated and raised."""
    # Test instantiation without message
    exc = EmptyDirNameException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Directory name cannot be empty"
    exc = EmptyDirNameException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(EmptyDirNameException):
        raise EmptyDirNameException("Test error message")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise EmptyDirNameException("Test error message")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise EmptyDirNameException("Test error message")


# LLM-generated content at query #21
#--------------------------

```python
def test_FailedHookException():
    """Test FailedHookException can be instantiated and raised."""
    # Test basic instantiation
    exc = FailedHookException("Test hook failed")
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == "Test hook failed"
    
    # Test with message
    message = "Hook script execution failed"
    exc = FailedHookException(message)
    assert exc.args[0] == message
    
    # Test raising the exception
    with pytest.raises(FailedHookException) as exc_info:
        raise FailedHookException("Hook failed during execution")
    assert "Hook failed during execution" in str(exc_info.value)
    
    # Test exception inheritance chain
    with pytest.raises(CookiecutterException):
        raise FailedHookException("Test")
    
    with pytest.raises(Exception):
        raise FailedHookException("Test")


# LLM-generated content at query #22
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test RepositoryCloneFailed exception can be instantiated and raised."""
    # Test basic instantiation
    exception = RepositoryCloneFailed("Test message")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"
    
    # Test instantiation with no message
    exception_no_msg = RepositoryCloneFailed()
    assert isinstance(exception_no_msg, RepositoryCloneFailed)
    
    # Test raising the exception
    with pytest.raises(RepositoryCloneFailed) as exc_info:
        raise RepositoryCloneFailed("Clone failed")
    assert "Clone failed" in str(exc_info.value)
    
    # Test exception args
    exception_args = RepositoryCloneFailed("arg1", "arg2")
    assert exception_args.args == ("arg1", "arg2")


# LLM-generated content at query #23
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test instantiation without arguments
    exc = VCSNotInstalled()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Git is not installed"
    exc = VCSNotInstalled(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial is not installed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("Version control system not found")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise VCSNotInstalled()


# LLM-generated content at query #24
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test UnknownTemplateDirException constructor and inheritance."""
    exception = UnknownTemplateDirException("Test message")
    
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"
    assert exception.args == ("Test message",)


def test_UnknownTemplateDirException_no_message():
    """Test UnknownTemplateDirException with no message."""
    exception = UnknownTemplateDirException()
    
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    assert exception.args == ()


def test_UnknownTemplateDirException_multiple_args():
    """Test UnknownTemplateDirException with multiple arguments."""
    exception = UnknownTemplateDirException("Error", "Details")
    
    assert isinstance(exception, CookiecutterException)
    assert exception.args == ("Error", "Details")


def test_UnknownTemplateDirException_can_be_raised():
    """Test that UnknownTemplateDirException can be raised and caught."""
    with pytest.raises(UnknownTemplateDirException) as exc_info:
        raise UnknownTemplateDirException("Template directory is ambiguous")
    
    assert str(exc_info.value) == "Template directory is ambiguous"


def test_UnknownTemplateDirException_can_be_caught_as_base_exception():
    """Test that UnknownTemplateDirException can be caught as CookiecutterException."""
    with pytest.raises(CookiecutterException):
        raise UnknownTemplateDirException("Test error")


# LLM-generated content at query #25
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test UnknownTemplateDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = UnknownTemplateDirException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Unable to determine which directory is the project template"
    exc = UnknownTemplateDirException(message)
    assert str(exc) == message
    assert exc.args == (message,)

    # Test raising the exception
    with pytest.raises(UnknownTemplateDirException):
        raise UnknownTemplateDirException("Test error")

    # Test exception inheritance chain
    with pytest.raises(CookiecutterException):
        raise UnknownTemplateDirException("Test error")

    with pytest.raises(Exception):
        raise UnknownTemplateDirException("Test error")


# LLM-generated content at query #26
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from unittest.mock import Mock
    
    # Create mock TemplateError
    mock_error = Mock()
    mock_error.message = "undefined variable 'foo'"
    
    # Create test context
    test_context = {"name": "test_project", "cookiecutter": {"project_name": "my_project"}}
    
    # Test message
    test_message = "Variable 'foo' is not defined"
    
    # Create exception instance
    exception = UndefinedVariableInTemplate(
        message=test_message,
        error=mock_error,
        context=test_context
    )
    
    # Verify attributes are set correctly
    assert exception.message == test_message
    assert exception.error == mock_error
    assert exception.context == test_context
    
    # Verify string representation
    expected_str = (
        f"{test_message}. "
        f"Error message: undefined variable 'foo'. "
        f"Context: {test_context}"
    )
    assert str(exception) == expected_str
    
    # Verify it's an instance of CookiecutterException
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #27
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test __str__ method of UndefinedVariableInTemplate exception."""
    from unittest.mock import Mock
    
    # Create a mock TemplateError
    mock_error = Mock()
    mock_error.message = "undefined variable 'foo'"
    
    # Create test context
    test_context = {"project_name": "test_project", "author": "test_author"}
    
    # Create the exception
    message = "Variable is not defined in the context"
    exception = UndefinedVariableInTemplate(
        message=message,
        error=mock_error,
        context=test_context
    )
    
    # Test __str__ output
    result = str(exception)
    
    # Verify the string representation contains all expected parts
    assert message in result
    assert "undefined variable 'foo'" in result
    assert "test_project" in result
    assert "test_author" in result
    assert "Error message:" in result
    assert "Context:" in result
    
    # Verify exact format
    expected = (
        f"{message}. "
        f"Error message: undefined variable 'foo'. "
        f"Context: {test_context}"
    )
    assert result == expected


# LLM-generated content at query #28
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = OutputDirExistsException()
    assert isinstance(exc, OutputDirExistsException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Output directory already exists"
    exc = OutputDirExistsException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Test error message")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise OutputDirExistsException("Test error message")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise OutputDirExistsException("Test error message")


# LLM-generated content at query #29
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test basic instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(message)
    assert str(exception_with_message) == message
    
    # Test raising the exception
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Test repository not found")
    
    # Test exception inheritance chain
    try:
        raise RepositoryNotFound("Repository does not exist")
    except CookiecutterException as e:
        assert isinstance(e, RepositoryNotFound)
        assert str(e) == "Repository does not exist"


# LLM-generated content at query #30
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from unittest.mock import Mock
    
    # Create a mock TemplateError
    mock_error = Mock()
    mock_error.message = "undefined variable 'foo'"
    
    # Test data
    message = "Variable is not defined"
    context = {"key1": "value1", "key2": "value2"}
    
    # Create exception instance
    exception = UndefinedVariableInTemplate(message, mock_error, context)
    
    # Assert attributes are set correctly
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Assert string representation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
    
    # Test with different context
    empty_context = {}
    exception2 = UndefinedVariableInTemplate("Another error", mock_error, empty_context)
    assert exception2.context == empty_context
    assert "Another error" in str(exception2)
    
    # Test that it's a subclass of CookiecutterException
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #31
#--------------------------

```python
def test_InvalidModeException():
    """Test InvalidModeException can be instantiated and raised."""
    # Test basic instantiation
    exc = InvalidModeException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Cannot use no_input and replay at the same time"
    exc = InvalidModeException(message)
    assert str(exc) == message
    
    # Test raising and catching
    with pytest.raises(InvalidModeException):
        raise InvalidModeException("Test error message")
    
    # Test catching as parent exception
    with pytest.raises(CookiecutterException):
        raise InvalidModeException("Test error message")
    
    # Test catching as base Exception
    with pytest.raises(Exception):
        raise InvalidModeException("Test error message")


# LLM-generated content at query #32
#--------------------------

```python
def test_MissingProjectDir():
    """Test MissingProjectDir exception initialization and inheritance."""
    # Test basic instantiation
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with message
    message = "Project directory not found"
    exception_with_msg = MissingProjectDir(message)
    assert str(exception_with_msg) == message
    
    # Test exception can be raised and caught
    with pytest.raises(MissingProjectDir):
        raise MissingProjectDir("Test error")
    
    # Test exception can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise MissingProjectDir("Test error")
    
    # Test exception can be caught as Exception
    with pytest.raises(Exception):
        raise MissingProjectDir("Test error")


# LLM-generated content at query #33
#--------------------------

```python
def test_FailedHookException():
    """Test FailedHookException can be instantiated and raised."""
    exception = FailedHookException("Hook script failed")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Hook script failed"
    
    # Test with additional arguments
    exception_with_args = FailedHookException("Hook failed", "extra_arg")
    assert str(exception_with_args) == "Hook failed"
    
    # Test raising the exception
    with pytest.raises(FailedHookException) as exc_info:
        raise FailedHookException("Test hook failure")
    assert "Test hook failure" in str(exc_info.value)


# LLM-generated content at query #34
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test ConfigDoesNotExistException can be instantiated and raised."""
    # Test instantiation with message
    exception = ConfigDoesNotExistException("Config file not found")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Config file not found"
    
    # Test instantiation without message
    exception_no_msg = ConfigDoesNotExistException()
    assert isinstance(exception_no_msg, CookiecutterException)
    assert str(exception_no_msg) == ""
    
    # Test raising the exception
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        raise ConfigDoesNotExistException("Missing config at /path/to/config")
    assert str(exc_info.value) == "Missing config at /path/to/config"


# LLM-generated content at query #35
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from unittest.mock import Mock
    
    # Create mock TemplateError
    mock_error = Mock()
    mock_error.message = "undefined variable 'foo'"
    
    # Create test context
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Test message
    test_message = "Variable 'foo' is not defined"
    
    # Initialize exception
    exc = UndefinedVariableInTemplate(test_message, mock_error, test_context)
    
    # Verify attributes are set correctly
    assert exc.message == test_message
    assert exc.error == mock_error
    assert exc.context == test_context
    
    # Verify string representation
    exc_str = str(exc)
    assert test_message in exc_str
    assert "undefined variable 'foo'" in exc_str
    assert str(test_context) in exc_str
    assert "Error message:" in exc_str
    assert "Context:" in exc_str
    
    # Test with different values
    mock_error2 = Mock()
    mock_error2.message = "undefined variable 'bar'"
    test_context2 = {"key": "value"}
    test_message2 = "Another error message"
    
    exc2 = UndefinedVariableInTemplate(test_message2, mock_error2, test_context2)
    exc_str2 = str(exc2)
    
    assert exc2.message == test_message2
    assert exc2.error == mock_error2
    assert exc2.context == test_context2
    assert test_message2 in exc_str2
    assert "undefined variable 'bar'" in exc_str2


# LLM-generated content at query #36
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = RepositoryNotFound()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Repository not found at the specified path"
    exc = RepositoryNotFound(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Test repository not found")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryNotFound("Another test message")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryNotFound("Generic exception test")


# LLM-generated content at query #37
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test NonTemplatedInputDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, NonTemplatedInputDirException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with a message
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert str(exception) == message
    
    # Test that it can be raised and caught
    with pytest.raises(NonTemplatedInputDirException):
        raise NonTemplatedInputDirException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise NonTemplatedInputDirException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise NonTemplatedInputDirException("Test error")


# LLM-generated content at query #38
#--------------------------

```python
def test_CookiecutterException():
    """Test CookiecutterException can be instantiated and raised."""
    # Test basic instantiation
    exc = CookiecutterException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, Exception)
    
    # Test raising and catching
    with pytest.raises(CookiecutterException) as exc_info:
        raise CookiecutterException("Test error")
    assert str(exc_info.value) == "Test error"
    
    # Test with no message
    exc_empty = CookiecutterException()
    assert isinstance(exc_empty, CookiecutterException)
    
    # Test inheritance
    assert issubclass(CookiecutterException, Exception)


# LLM-generated content at query #39
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test UnknownTemplateDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = UnknownTemplateDirException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Cannot determine template directory"
    exc = UnknownTemplateDirException(message)
    assert str(exc) == message
    assert exc.args == (message,)

    # Test instantiation with multiple arguments
    exc = UnknownTemplateDirException("Error", "Additional info")
    assert exc.args == ("Error", "Additional info")

    # Test raising the exception
    with pytest.raises(UnknownTemplateDirException):
        raise UnknownTemplateDirException("Test error")

    # Test catching as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownTemplateDirException("Test error")

    # Test catching as Exception
    with pytest.raises(Exception):
        raise UnknownTemplateDirException("Test error")


# LLM-generated content at query #40
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException can be instantiated and raised."""
    # Test basic instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with message
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    assert str(exception) == message
    
    # Test raising and catching
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Test error message")
    
    # Test with multiple arguments
    exception = OutputDirExistsException("Error", "Additional info")
    assert isinstance(exception, OutputDirExistsException)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = RepositoryNotFound()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "The specified cookiecutter repository doesn't exist"
    exc = RepositoryNotFound(message)
    assert str(exc) == message
    
    # Test raising the exception
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Repository not found at path")
    
    # Test catching as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryNotFound("Repository not found")
    
    # Test exception message is preserved
    try:
        raise RepositoryNotFound("Custom error message")
    except RepositoryNotFound as e:
        assert "Custom error message" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException can be instantiated and raised."""
    # Test basic instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    assert str(exception) == message
    assert exception.args == (message,)
    
    # Test raising the exception
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Directory exists at /path/to/output")
    
    # Test raising and catching with message verification
    try:
        raise OutputDirExistsException("Test output directory exists")
    except OutputDirExistsException as e:
        assert "Test output directory exists" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidZipRepository():
    """Test InvalidZipRepository exception can be instantiated and raised."""
    # Test basic instantiation
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Invalid zip repository"
    exc_with_msg = InvalidZipRepository(message)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    with pytest.raises(InvalidZipRepository):
        raise InvalidZipRepository("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise InvalidZipRepository("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise InvalidZipRepository("Test error")


# LLM-generated content at query #4
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test UnknownTemplateDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert str(exception) == message
    assert exception.args == (message,)
    
    # Test that it can be raised and caught
    with pytest.raises(UnknownTemplateDirException):
        raise UnknownTemplateDirException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownTemplateDirException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise UnknownTemplateDirException("Test error")


# LLM-generated content at query #5
#--------------------------

```python
def test_UnknownRepoType():
    """Test UnknownRepoType exception can be instantiated and raised."""
    # Test instantiation with a message
    exception = UnknownRepoType("Test message")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"

    # Test instantiation without a message
    exception_empty = UnknownRepoType()
    assert isinstance(exception_empty, CookiecutterException)

    # Test that it can be raised and caught
    with pytest.raises(UnknownRepoType) as exc_info:
        raise UnknownRepoType("Repository type unknown")
    assert str(exc_info.value) == "Repository type unknown"

    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownRepoType("Another test")

    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise UnknownRepoType("Generic exception test")


# LLM-generated content at query #6
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test RepositoryCloneFailed exception."""
    # Test instantiation with no arguments
    exc = RepositoryCloneFailed()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Failed to clone the repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    
    # Test instantiation with multiple arguments
    exc = RepositoryCloneFailed("Error", "Additional info")
    assert isinstance(exc, RepositoryCloneFailed)
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryCloneFailed):
        raise RepositoryCloneFailed("Clone failed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryCloneFailed("Clone failed")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryCloneFailed("Clone failed")


# LLM-generated content at query #7
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test ConfigDoesNotExistException can be instantiated and raised."""
    # Test instantiation with a message
    exception = ConfigDoesNotExistException("Config file not found")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Config file not found"
    
    # Test instantiation without a message
    exception_no_msg = ConfigDoesNotExistException()
    assert isinstance(exception_no_msg, CookiecutterException)
    
    # Test that it can be raised and caught
    with pytest.raises(ConfigDoesNotExistException):
        raise ConfigDoesNotExistException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise ConfigDoesNotExistException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise ConfigDoesNotExistException("Test error")


# LLM-generated content at query #8
#--------------------------

```python
def test_InvalidConfiguration():
    """Test InvalidConfiguration exception."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Invalid YAML configuration"
    exc = InvalidConfiguration(message)
    assert str(exc) == message
    
    # Test with multiple arguments
    exc = InvalidConfiguration("Config error", "additional info")
    assert "Config error" in str(exc)
    
    # Test exception can be raised and caught
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration("Test error message")
    
    # Test exception can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise InvalidConfiguration("Test error")
    
    # Test exception can be caught as Exception
    with pytest.raises(Exception):
        raise InvalidConfiguration("Test error")


# LLM-generated content at query #9
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test UnknownTemplateDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = UnknownTemplateDirException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Ambiguous template directory"
    exc = UnknownTemplateDirException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(UnknownTemplateDirException):
        raise UnknownTemplateDirException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownTemplateDirException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise UnknownTemplateDirException("Test error")


# LLM-generated content at query #10
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException can be instantiated and raised."""
    # Test instantiation with a message
    exception = OutputDirExistsException("Output directory already exists")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Output directory already exists"
    
    # Test instantiation without a message
    exception_no_msg = OutputDirExistsException()
    assert isinstance(exception_no_msg, CookiecutterException)
    assert str(exception_no_msg) == ""
    
    # Test raising the exception
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Test error")
    
    # Test exception with multiple arguments
    exception_multi = OutputDirExistsException("Error", "with", "multiple", "args")
    assert isinstance(exception_multi, CookiecutterException)


# LLM-generated content at query #11
#--------------------------

```python
def test_EmptyDirNameException():
    """Test EmptyDirNameException constructor and inheritance."""
    # Test that EmptyDirNameException can be instantiated
    exception = EmptyDirNameException("Directory name is empty")
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test that the message is properly stored
    assert str(exception) == "Directory name is empty"
    assert exception.args == ("Directory name is empty",)
    
    # Test with no message
    exception_no_msg = EmptyDirNameException()
    assert isinstance(exception_no_msg, EmptyDirNameException)
    assert str(exception_no_msg) == ""
    
    # Test with multiple arguments
    exception_multi = EmptyDirNameException("Error", "Additional info")
    assert exception_multi.args == ("Error", "Additional info")


# LLM-generated content at query #12
#--------------------------

```python
def test_InvalidConfiguration():
    """Test InvalidConfiguration exception can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Configuration file is not valid YAML"
    exc = InvalidConfiguration(message)
    assert str(exc) == message
    
    # Test raising the exception
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration("Invalid config")
    
    # Test raising with no message
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration()
    
    # Test exception inheritance chain
    with pytest.raises(CookiecutterException):
        raise InvalidConfiguration("Test message")
    
    with pytest.raises(Exception):
        raise InvalidConfiguration("Test message")


# LLM-generated content at query #13
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test basic instantiation
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert str(exc_with_msg) == message
    
    # Test raising the exception
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial is not installed")
    
    # Test exception message is preserved
    try:
        raise VCSNotInstalled("Version control system not found")
    except VCSNotInstalled as e:
        assert str(e) == "Version control system not found"


# LLM-generated content at query #14
#--------------------------

```python
def test_FailedHookException():
    """Test FailedHookException can be instantiated and raised."""
    exception = FailedHookException("Hook script failed")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Hook script failed"
    
    with pytest.raises(FailedHookException) as exc_info:
        raise FailedHookException("Test hook failure")
    
    assert str(exc_info.value) == "Test hook failure"


# LLM-generated content at query #15
#--------------------------

```python
def test_ContextDecodingException():
    """Test ContextDecodingException can be instantiated and raised."""
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, ContextDecodingException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Failed to decode context JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    
    # Test raising the exception
    with pytest.raises(ContextDecodingException):
        raise ContextDecodingException("Test error message")
    
    # Test raising with message and catching
    try:
        raise ContextDecodingException("Context JSON decode error")
    except ContextDecodingException as e:
        assert str(e) == "Context JSON decode error"


# LLM-generated content at query #16
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial is not installed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("VCS not found")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise VCSNotInstalled()


# LLM-generated content at query #17
#--------------------------

```python
def test_CookiecutterException():
    """Test CookiecutterException can be instantiated and raised."""
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with message
    message = "Test error message"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert exc.args == (message,)
    
    # Test that it can be raised and caught
    with pytest.raises(CookiecutterException):
        raise CookiecutterException("Test exception")
    
    # Test inheritance
    assert issubclass(CookiecutterException, Exception)
    
    # Test multiple arguments
    exc = CookiecutterException("arg1", "arg2")
    assert exc.args == ("arg1", "arg2")


# LLM-generated content at query #18
#--------------------------

```python
def test_MissingProjectDir():
    """Test MissingProjectDir exception can be instantiated and raised."""
    # Test basic instantiation
    exc = MissingProjectDir()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Project directory not found"
    exc = MissingProjectDir(message)
    assert str(exc) == message
    
    # Test raising the exception
    with pytest.raises(MissingProjectDir):
        raise MissingProjectDir("Test error message")
    
    # Test exception hierarchy
    with pytest.raises(CookiecutterException):
        raise MissingProjectDir("Test error message")


# LLM-generated content at query #19
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test ConfigDoesNotExistException can be instantiated and raised."""
    # Test instantiation with a message
    exception = ConfigDoesNotExistException("Config file not found")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Config file not found"
    
    # Test instantiation without a message
    exception_no_msg = ConfigDoesNotExistException()
    assert isinstance(exception_no_msg, CookiecutterException)
    assert str(exception_no_msg) == ""
    
    # Test that it can be raised and caught
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        raise ConfigDoesNotExistException("Path does not exist: /invalid/path")
    
    assert str(exc_info.value) == "Path does not exist: /invalid/path"
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise ConfigDoesNotExistException("Config error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise ConfigDoesNotExistException("Config error")


# LLM-generated content at query #20
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test NonTemplatedInputDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = NonTemplatedInputDirException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Input directory is not templated"
    exc = NonTemplatedInputDirException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(NonTemplatedInputDirException):
        raise NonTemplatedInputDirException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise NonTemplatedInputDirException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise NonTemplatedInputDirException("Test error")


# LLM-generated content at query #21
#--------------------------

```python
def test_ContextDecodingException():
    """Test ContextDecodingException can be instantiated and raised."""
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, ContextDecodingException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with message
    message = "Failed to decode context JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert exception.args == (message,)
    
    # Test raising and catching
    with pytest.raises(ContextDecodingException):
        raise ContextDecodingException("Test error message")
    
    # Test catching as parent exception
    with pytest.raises(CookiecutterException):
        raise ContextDecodingException("Test error message")
    
    # Test catching as generic Exception
    with pytest.raises(Exception):
        raise ContextDecodingException("Test error message")


# LLM-generated content at query #22
#--------------------------

```python
def test_InvalidConfiguration():
    """Test InvalidConfiguration exception initialization and inheritance."""
    # Test that InvalidConfiguration can be instantiated
    exc = InvalidConfiguration()
    assert isinstance(exc, InvalidConfiguration)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with a message
    message = "Invalid YAML configuration"
    exc_with_msg = InvalidConfiguration(message)
    assert str(exc_with_msg) == message
    assert exc_with_msg.args == (message,)
    
    # Test with multiple arguments
    exc_multi = InvalidConfiguration("Error", "Details")
    assert exc_multi.args == ("Error", "Details")
    
    # Test that it can be raised and caught
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration("Config is not valid YAML")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise InvalidConfiguration("Badly constructed config")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise InvalidConfiguration("Generic error")


# LLM-generated content at query #23
#--------------------------

```python
def test_EmptyDirNameException():
    """Test EmptyDirNameException can be instantiated and raised."""
    # Test instantiation without arguments
    exc = EmptyDirNameException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Directory name cannot be empty"
    exc = EmptyDirNameException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(EmptyDirNameException):
        raise EmptyDirNameException("Test error message")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise EmptyDirNameException("Test error message")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise EmptyDirNameException("Test error message")


# LLM-generated content at query #24
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from jinja2 import UndefinedError
    
    # Create a mock TemplateError
    template_error = UndefinedError("variable 'foo' is undefined")
    message = "Variable is not defined"
    context = {"key1": "value1", "key2": "value2"}
    
    # Test initialization
    exception = UndefinedVariableInTemplate(message, template_error, context)
    
    assert exception.message == message
    assert exception.error == template_error
    assert exception.context == context
    
    # Test string representation
    str_repr = str(exception)
    assert message in str_repr
    assert "Error message:" in str_repr
    assert "Context:" in str_repr
    assert str(context) in str_repr
    
    # Test that it's a subclass of CookiecutterException
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with empty context
    empty_context = {}
    exception_empty = UndefinedVariableInTemplate(message, template_error, empty_context)
    assert exception_empty.context == empty_context
    
    # Test with different message
    different_message = "Different error occurred"
    exception_diff = UndefinedVariableInTemplate(different_message, template_error, context)
    assert exception_diff.message == different_message
    assert different_message in str(exception_diff)


# LLM-generated content at query #25
#--------------------------

```python
def test_InvalidModeException():
    """Test InvalidModeException can be instantiated and raised."""
    # Test instantiation with a message
    exception = InvalidModeException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"
    
    # Test raising the exception
    with pytest.raises(InvalidModeException) as exc_info:
        raise InvalidModeException("no_input and replay cannot both be True")
    assert "no_input and replay cannot both be True" in str(exc_info.value)
    
    # Test with empty message
    exception_empty = InvalidModeException("")
    assert str(exception_empty) == ""
    
    # Test inheritance chain
    with pytest.raises(CookiecutterException):
        raise InvalidModeException("Test")


# LLM-generated content at query #26
#--------------------------

```python
def test_InvalidConfiguration():
    """Test InvalidConfiguration exception can be instantiated and raised."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Invalid YAML in config file"
    exc = InvalidConfiguration(message)
    assert str(exc) == message
    
    # Test raising the exception
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration()
    
    # Test raising with message
    with pytest.raises(InvalidConfiguration) as exc_info:
        raise InvalidConfiguration("Config file is not valid YAML")
    assert str(exc_info.value) == "Config file is not valid YAML"
    
    # Test that it's a subclass of CookiecutterException
    assert issubclass(InvalidConfiguration, CookiecutterException)


# LLM-generated content at query #27
#--------------------------

```python
def test_InvalidModeException():
    """Test InvalidModeException can be instantiated and raised."""
    exception = InvalidModeException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == "Test message"
    
    with pytest.raises(InvalidModeException) as exc_info:
        raise InvalidModeException("Both no_input and replay cannot be True")
    
    assert "Both no_input and replay cannot be True" in str(exc_info.value)


# LLM-generated content at query #28
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test NonTemplatedInputDirException can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = NonTemplatedInputDirException()
    assert isinstance(exc, NonTemplatedInputDirException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Input directory is not templated"
    exc = NonTemplatedInputDirException(message)
    assert str(exc) == message
    assert exc.args == (message,)
    
    # Test raising the exception
    with pytest.raises(NonTemplatedInputDirException):
        raise NonTemplatedInputDirException("Test error message")
    
    # Test raising and catching with parent class
    with pytest.raises(CookiecutterException):
        raise NonTemplatedInputDirException("Test error message")
    
    # Test raising and catching with Exception
    with pytest.raises(Exception):
        raise NonTemplatedInputDirException("Test error message")


# LLM-generated content at query #29
#--------------------------

```python
def test_UnknownRepoType():
    """Test UnknownRepoType exception can be instantiated and raised."""
    # Test basic instantiation
    exc = UnknownRepoType()
    assert isinstance(exc, UnknownRepoType)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Unable to determine repository type"
    exc_with_msg = UnknownRepoType(message)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    with pytest.raises(UnknownRepoType):
        raise UnknownRepoType("Test error message")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise UnknownRepoType("Test error message")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise UnknownRepoType("Test error message")


# LLM-generated content at query #30
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test basic instantiation
    exc = VCSNotInstalled()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial not found")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("Version control system unavailable")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise VCSNotInstalled()


# LLM-generated content at query #31
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test instantiation with a message
    exc = RepositoryNotFound("Test repository not found")
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == "Test repository not found"
    
    # Test instantiation without a message
    exc_empty = RepositoryNotFound()
    assert isinstance(exc_empty, CookiecutterException)
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Repository does not exist")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryNotFound("Repository error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryNotFound("Generic error")
    
    # Test args attribute
    exc_with_args = RepositoryNotFound("arg1", "arg2")
    assert exc_with_args.args == ("arg1", "arg2")


# LLM-generated content at query #32
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test __str__ method of UndefinedVariableInTemplate."""
    from unittest.mock import Mock
    
    # Create a mock TemplateError
    mock_error = Mock()
    mock_error.message = "undefined variable 'foo'"
    
    # Create a test context
    test_context = {"project_name": "my_project", "author": "John Doe"}
    
    # Create the exception
    message = "Variable is undefined"
    exception = UndefinedVariableInTemplate(message, mock_error, test_context)
    
    # Test __str__ output
    result = str(exception)
    
    assert "Variable is undefined" in result
    assert "undefined variable 'foo'" in result
    assert "project_name" in result
    assert "my_project" in result
    assert "Error message:" in result
    assert "Context:" in result


def test_UndefinedVariableInTemplate___str__empty_context():
    """Test __str__ method with empty context."""
    from unittest.mock import Mock
    
    mock_error = Mock()
    mock_error.message = "undefined variable 'bar'"
    
    message = "Template variable error"
    exception = UndefinedVariableInTemplate(message, mock_error, {})
    
    result = str(exception)
    
    assert "Template variable error" in result
    assert "undefined variable 'bar'" in result
    assert "{}" in result


def test_UndefinedVariableInTemplate___str__complex_context():
    """Test __str__ method with complex context."""
    from unittest.mock import Mock
    
    mock_error = Mock()
    mock_error.message = "undefined variable 'nested_var'"
    
    test_context = {
        "simple": "value",
        "nested": {"key": "value"},
        "list": [1, 2, 3]
    }
    
    message = "Complex context error"
    exception = UndefinedVariableInTemplate(message, mock_error, test_context)
    
    result = str(exception)
    
    assert "Complex context error" in result
    assert "undefined variable 'nested_var'" in result
    assert "simple" in result
    assert "nested" in result
    assert "list" in result


# LLM-generated content at query #33
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test RepositoryCloneFailed exception can be instantiated and raised."""
    # Test instantiation with no arguments
    exc = RepositoryCloneFailed()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Failed to clone the repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryCloneFailed):
        raise RepositoryCloneFailed("Clone operation failed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryCloneFailed("Clone operation failed")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryCloneFailed("Clone operation failed")


# LLM-generated content at query #34
#--------------------------

```python
def test_RepositoryNotFound():
    """Test RepositoryNotFound exception can be instantiated and raised."""
    # Test basic instantiation
    exc = RepositoryNotFound()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with a message
    message = "The repository could not be found"
    exc_with_message = RepositoryNotFound(message)
    assert str(exc_with_message) == message
    
    # Test that it can be raised and caught
    with pytest.raises(RepositoryNotFound):
        raise RepositoryNotFound("Repository not found at the specified path")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise RepositoryNotFound("Repository error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise RepositoryNotFound("Generic exception")


# LLM-generated content at query #35
#--------------------------

```python
def test_InvalidZipRepository():
    """Test InvalidZipRepository exception can be instantiated and raised."""
    # Test basic instantiation
    exc = InvalidZipRepository()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Invalid zip repository"
    exc = InvalidZipRepository(message)
    assert str(exc) == message
    
    # Test raising the exception
    with pytest.raises(InvalidZipRepository):
        raise InvalidZipRepository("test message")
    
    # Test catching as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise InvalidZipRepository("test message")
    
    # Test catching as Exception
    with pytest.raises(Exception):
        raise InvalidZipRepository("test message")


# LLM-generated content at query #36
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    """Test UndefinedVariableInTemplate exception initialization and string representation."""
    from jinja2 import TemplateError
    
    # Create a mock TemplateError
    template_error = TemplateError("Variable 'foo' is undefined")
    template_error.message = "Variable 'foo' is undefined"
    
    # Test data
    message = "Error in template rendering"
    context = {"key1": "value1", "key2": "value2"}
    
    # Create the exception
    exc = UndefinedVariableInTemplate(message, template_error, context)
    
    # Verify attributes are set correctly
    assert exc.message == message
    assert exc.error == template_error
    assert exc.context == context
    
    # Verify string representation
    exc_str = str(exc)
    assert message in exc_str
    assert "Variable 'foo' is undefined" in exc_str
    assert "key1" in exc_str
    assert "value1" in exc_str
    
    # Verify it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


def test_UndefinedVariableInTemplate_with_empty_context():
    """Test UndefinedVariableInTemplate with empty context."""
    from jinja2 import TemplateError
    
    template_error = TemplateError("undefined variable")
    template_error.message = "undefined variable"
    
    message = "Template error occurred"
    context = {}
    
    exc = UndefinedVariableInTemplate(message, template_error, context)
    
    assert exc.message == message
    assert exc.context == {}
    assert "Template error occurred" in str(exc)


def test_UndefinedVariableInTemplate_with_complex_context():
    """Test UndefinedVariableInTemplate with complex nested context."""
    from jinja2 import TemplateError
    
    template_error = TemplateError("Complex error")
    template_error.message = "Complex error message"
    
    message = "Complex template rendering failed"
    context = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "string": "test"
    }
    
    exc = UndefinedVariableInTemplate(message, template_error, context)
    
    assert exc.context == context
    exc_str = str(exc)
    assert "Complex template rendering failed" in exc_str
    assert "Complex error message" in exc_str


# LLM-generated content at query #37
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test basic instantiation
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test with message
    message = "Git is not installed"
    exc_with_message = VCSNotInstalled(message)
    assert str(exc_with_message) == message
    
    # Test raising the exception
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial is not installed")
    
    # Test exception hierarchy
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("VCS not found")
    
    # Test with multiple arguments
    exc_multi = VCSNotInstalled("arg1", "arg2")
    assert exc_multi.args == ("arg1", "arg2")


# LLM-generated content at query #38
#--------------------------

```python
def test_OutputDirExistsException():
    """Test OutputDirExistsException constructor and inheritance."""
    # Test that OutputDirExistsException can be instantiated
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with a message
    message = "Output directory already exists"
    exception_with_msg = OutputDirExistsException(message)
    assert str(exception_with_msg) == message
    assert exception_with_msg.args == (message,)
    
    # Test with multiple arguments
    exception_multi_args = OutputDirExistsException("Error", "Details")
    assert exception_multi_args.args == ("Error", "Details")
    
    # Test that it can be raised and caught
    with pytest.raises(OutputDirExistsException):
        raise OutputDirExistsException("Test error")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise OutputDirExistsException("Test error")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise OutputDirExistsException("Test error")


# LLM-generated content at query #39
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test instantiation without arguments
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert isinstance(exc_with_msg, VCSNotInstalled)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial not found")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("Version control system unavailable")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise VCSNotInstalled()


# LLM-generated content at query #40
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception can be instantiated and raised."""
    # Test instantiation with no message
    exc = VCSNotInstalled()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    
    # Test instantiation with a message
    message = "Git is not installed"
    exc = VCSNotInstalled(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    with pytest.raises(VCSNotInstalled):
        raise VCSNotInstalled("Mercurial is not installed")
    
    # Test that it can be caught as CookiecutterException
    with pytest.raises(CookiecutterException):
        raise VCSNotInstalled("VCS not found")
    
    # Test that it can be caught as Exception
    with pytest.raises(Exception):
        raise VCSNotInstalled("Version control system unavailable")


