####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #2
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid config file")
    assert str(exception) == "Invalid config file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #3
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #4
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #5
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no arguments
    exc = ContextDecodingException()
    assert str(exc) == ""
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #6
#--------------------------

```python
def test_InvalidConfiguration():
    # Test that InvalidConfiguration can be instantiated with a message
    message = "Invalid configuration provided"
    exception = InvalidConfiguration(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #7
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #8
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic instantiation
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Config file not found"
    exc_with_msg = ConfigDoesNotExistException(message)
    assert str(exc_with_msg) == message

    # Test default message is empty
    assert str(exc) == ""


# LLM-generated content at query #9
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #10
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #11
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #12
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #13
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #14
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not defined"
    error = Mock()
    error.message = "Template error occurred"
    context = {"key": "value"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable not defined. "
        "Error message: Template error occurred. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #15
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no arguments
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""
    assert isinstance(exc_empty, CookiecutterException)


# LLM-generated content at query #16
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownRepoType)
    assert str(exception) == ""


# LLM-generated content at query #17
#--------------------------

```python
def test_RepositoryNotFound():
    # Test default message
    exc = RepositoryNotFound()
    assert str(exc) == ""

    # Test custom message
    custom_msg = "Custom error message"
    exc = RepositoryNotFound(custom_msg)
    assert str(exc) == custom_msg


# LLM-generated content at query #18
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #19
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #20
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Create a mock TemplateError object
    class MockTemplateError:
        def __init__(self, message):
            self.message = message

    # Create an instance of UndefinedVariableInTemplate
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"bar": "baz"}
    exception = UndefinedVariableInTemplate(
        message="Undefined variable in template",
        error=error,
        context=context
    )

    # Test the __str__ method
    result = str(exception)
    expected = (
        "Undefined variable in template. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'bar': 'baz'}"
    )
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #22
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test message")
    except InvalidConfiguration as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #23
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #24
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #25
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    assert exc.message == message


# LLM-generated content at query #26
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #27
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Test message")
    except UnknownRepoType as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #28
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not defined"
    error = Exception("Error in template")
    context = {"key": "value"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #29
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #30
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #31
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #32
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #33
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #34
#--------------------------

```python
def test_UnknownRepoType():
    # Test that UnknownRepoType is an instance of CookiecutterException
    exception = UnknownRepoType("Unknown repository type")
    assert isinstance(exception, CookiecutterException)

    # Test that the exception message is set correctly
    assert str(exception) == "Unknown repository type"

    # Test that the exception can be raised and caught
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
    except Exception:
        assert False, "UnknownRepoType should be catchable as itself"


# LLM-generated content at query #35
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #36
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Test message"
    error = type('MockTemplateError', (), {'message': 'Test error message'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        "Test message. "
        "Error message: Test error message. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #37
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #38
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #39
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context file"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #40
#--------------------------

```python
def test_CookiecutterException():
    """Test that CookiecutterException can be instantiated."""
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)


# LLM-generated content at query #41
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #42
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidZipRepository)
    assert str(exception) == ""


# LLM-generated content at query #43
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    custom_msg = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(custom_msg)
    assert str(exc_with_msg) == custom_msg

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #44
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #45
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exc = FailedHookException("Hook failed")
    assert str(exc) == "Hook failed"
    assert isinstance(exc, CookiecutterException)

    # Test with additional context
    exc_with_context = FailedHookException("Hook failed", "script.py", "pre_gen_project")
    assert str(exc_with_context) == "Hook failed"
    assert exc_with_context.script == "script.py"
    assert exc_with_context.hook_type == "pre_gen_project"


# LLM-generated content at query #46
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #47
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = TemplateError("Template error occurred")
    context = {"bar": "baz"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #48
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #49
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert exception.message == message


# LLM-generated content at query #50
#--------------------------

```python
def test_RepositoryNotFound():
    # Test default constructor
    exc = RepositoryNotFound()
    assert str(exc) == ""

    # Test constructor with custom message
    custom_message = "Custom error message"
    exc = RepositoryNotFound(custom_message)
    assert str(exc) == custom_message


# LLM-generated content at query #51
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #52
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #53
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #54
#--------------------------

```python
def test_UnknownRepoType():
    """Test the constructor of UnknownRepoType exception."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #55
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #56
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound."""
    try:
        raise RepositoryNotFound("Repository not found")
    except RepositoryNotFound as e:
        assert str(e) == "Repository not found"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #57
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #58
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert exc.args == (message,)


# LLM-generated content at query #59
#--------------------------

```python
def test_CookiecutterException():
    """Test the constructor of CookiecutterException."""
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with a message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #60
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #61
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #62
#--------------------------

```python
def test_MissingProjectDir():
    # Test that MissingProjectDir can be instantiated
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #63
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #64
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    try:
        raise InvalidConfiguration("Invalid configuration provided")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration provided"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #65
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed."""
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #66
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Setup test data
    message = "Test message"
    error = Exception("Test error")
    context = {"key": "value"}

    # Create instance
    exception = UndefinedVariableInTemplate(message, error, context)

    # Verify attributes
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context

    # Verify string representation
    expected_str = (
        f"{message}. "
        f"Error message: {error}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str


# LLM-generated content at query #67
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    message = "Unknown extension"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #68
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert issubclass(CookiecutterException, Exception)


# LLM-generated content at query #69
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #70
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #71
#--------------------------

```python
def test_InvalidModeException():
    """Test the InvalidModeException constructor."""
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #72
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation
    exc = UnknownExtension()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, UnknownExtension)
    assert str(exc) == ""

    # Test with a custom message
    message = "Custom error message"
    exc_with_msg = UnknownExtension(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #73
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, ConfigDoesNotExistException)
    assert str(exception) == ""


# LLM-generated content at query #74
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)

    # Test with no arguments
    exception_empty = ContextDecodingException()
    assert str(exception_empty) == ""
    assert isinstance(exception_empty, CookiecutterException)


# LLM-generated content at query #75
#--------------------------

```python
def test_MissingProjectDir():
    """Test the constructor of MissingProjectDir exception."""
    exception = MissingProjectDir("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #76
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Invalid mode provided")
    assert str(exception) == "Invalid mode provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #77
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #78
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #79
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #80
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = TemplateError("Undefined variable 'foo'")
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    message = "Custom error message"
    exc_with_msg = RepositoryNotFound(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #2
#--------------------------

```python
def test_OutputDirExistsException():
    """Test the constructor of OutputDirExistsException."""
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #4
#--------------------------

```python
def test_UnknownTemplateDirException():
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #5
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #6
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    custom_msg = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(custom_msg)
    assert str(exc_with_msg) == custom_msg

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #7
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #8
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #9
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test NonTemplatedInputDirException constructor."""
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #10
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #11
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #12
#--------------------------

```python
def test_OutputDirExistsException():
    # Test that OutputDirExistsException can be instantiated and is a subclass of CookiecutterException
    exception = OutputDirExistsException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #13
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #14
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test the constructor of ConfigDoesNotExistException."""
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #15
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Create a mock TemplateError
    class MockTemplateError:
        def __init__(self, message):
            self.message = message

    # Create an instance of UndefinedVariableInTemplate
    error = UndefinedVariableInTemplate(
        message="Variable 'foo' is undefined",
        error=MockTemplateError("Template rendering failed"),
        context={"bar": "baz"}
    )

    # Test the __str__ method
    assert str(error) == (
        "Variable 'foo' is undefined. "
        "Error message: Template rendering failed. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #16
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)


# LLM-generated content at query #17
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #18
#--------------------------

```python
def test_InvalidModeException():
    """Test the InvalidModeException constructor."""
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #19
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #20
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #21
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception constructor."""
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #22
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #23
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #24
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #25
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Test message"
    error = Exception("Test error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #26
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed exception."""
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with a custom message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #27
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed."""
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #28
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #29
#--------------------------

```python
def test_RepositoryNotFound():
    """Test that RepositoryNotFound exception is correctly initialized."""
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #30
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = Exception("Template error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #31
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #32
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = TemplateError("Template rendering error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #33
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #34
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = TemplateError("Template error occurred")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #35
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #36
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    # Test basic instantiation
    exc = CookiecutterException("Test message")
    assert str(exc) == "Test message"

    # Test instantiation without message
    exc = CookiecutterException()
    assert str(exc) == ""

    # Test that it's a subclass of Exception
    assert isinstance(exc, Exception)


# LLM-generated content at query #37
#--------------------------

```python
def test_MissingProjectDir():
    # Test that MissingProjectDir can be instantiated with a custom message
    message = "Custom error message"
    exception = MissingProjectDir(message)
    assert str(exception) == message

    # Test that MissingProjectDir is an instance of CookiecutterException
    assert isinstance(exception, CookiecutterException)

    # Test that MissingProjectDir is an instance of Exception
    assert isinstance(exception, Exception)


# LLM-generated content at query #38
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #39
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = TemplateError("Template error occurred")
    context = {"bar": "baz"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #40
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception initialization."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #41
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #42
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #43
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""
    assert isinstance(exc_empty, CookiecutterException)


# LLM-generated content at query #44
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #45
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #46
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    hook_name = "pre_gen_project"
    exception = FailedHookException(message, hook_name)
    assert str(exception) == f"{message}: {hook_name}"
    assert exception.message == message
    assert exception.hook_name == hook_name


# LLM-generated content at query #47
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #48
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the constructor of VCSNotInstalled exception."""
    # Test default constructor
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    message = "Git is not installed"
    exc = VCSNotInstalled(message)
    assert str(exc) == message


# LLM-generated content at query #49
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #50
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #51
#--------------------------

```python
def test_OutputDirExistsException():
    """Test that OutputDirExistsException can be instantiated."""
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #52
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation
    exc = OutputDirExistsException()
    assert isinstance(exc, OutputDirExistsException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with custom message
    message = "Custom error message"
    exc_with_msg = OutputDirExistsException(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #53
#--------------------------

```python
def test_UnknownRepoType():
    """Test the constructor of UnknownRepoType exception."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #54
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Create a mock TemplateError object
    class MockTemplateError:
        def __init__(self, message):
            self.message = message

    # Test data
    message = "Variable not defined"
    error = MockTemplateError("Template rendering failed")
    context = {"key": "value"}

    # Create an instance of UndefinedVariableInTemplate
    exception = UndefinedVariableInTemplate(message, error, context)

    # Expected string representation
    expected_str = (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )

    # Assert the string representation matches the expected output
    assert str(exception) == expected_str


# LLM-generated content at query #55
#--------------------------

```python
def test_InvalidModeException():
    """Test the constructor of InvalidModeException."""
    exception = InvalidModeException("Test message")
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #56
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test the constructor of NonTemplatedInputDirException."""
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #57
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #58
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #59
#--------------------------

```python
def test_InvalidModeException():
    try:
        raise InvalidModeException("Test message")
    except InvalidModeException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #60
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #61
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #62
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #63
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #64
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = Exception("Error in template")
    context = {"bar": "baz"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #65
#--------------------------

```python
def test_FailedHookException():
    exception = FailedHookException("Hook failed")
    assert str(exception) == "Hook failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #66
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #67
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #68
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #69
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation
    exc = OutputDirExistsException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc = OutputDirExistsException()
    assert str(exc) == ""
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #70
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #71
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation
    exc = UnknownExtension("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc_no_msg = UnknownExtension()
    assert str(exc_no_msg) == ""
    assert isinstance(exc_no_msg, CookiecutterException)

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #72
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #73
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #74
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = TemplateError("Template error occurred")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #75
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType can be instantiated."""
    try:
        raise UnknownRepoType("Test message")
    except UnknownRepoType as e:
        assert str(e) == "Test message"


