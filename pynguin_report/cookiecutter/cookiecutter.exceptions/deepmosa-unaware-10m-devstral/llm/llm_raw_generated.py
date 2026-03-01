####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = Mock()
    error.message = "Variable 'foo' is not defined"
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is undefined. "
        "Error message: Variable 'foo' is not defined. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_NonTemplatedInputDirException():
    # Test default constructor
    exc = NonTemplatedInputDirException()
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    message = "Input directory is not templated"
    exc = NonTemplatedInputDirException(message)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message


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
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is not defined"
    error = type('MockTemplateError', (), {'message': 'Template rendering error'})()
    context = {'bar': 'baz'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is not defined. "
        "Error message: Template rendering error. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #5
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    with pytest.raises(InvalidConfiguration):
        raise InvalidConfiguration("Test message")


# LLM-generated content at query #6
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #7
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test message")
    except InvalidConfiguration as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #8
#--------------------------

```python
def test_InvalidModeException():
    """Test the constructor of InvalidModeException."""
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #9
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert exception.message == message


# LLM-generated content at query #10
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #11
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid config")
    assert str(exception) == "Invalid config"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #12
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #13
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid config")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid config"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #14
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of the UnknownExtension exception."""
    message = "Unknown extension error"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #15
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid config file")
    assert str(exception) == "Invalid config file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #16
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #17
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #18
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #19
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Test message"
    error = Mock()
    error.message = "Test error message"
    context = {"key": "value"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Test message. "
        "Error message: Test error message. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #24
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with a custom message
    custom_message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(custom_message)
    assert str(exc_with_msg) == custom_message

    # Test default message is empty
    assert str(exc) == ""


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
def test_VCSNotInstalled():
    # Test default constructor
    exc = VCSNotInstalled()
    assert str(exc) == ""

    # Test constructor with message
    exc = VCSNotInstalled("Git is not installed")
    assert str(exc) == "Git is not installed"


# LLM-generated content at query #27
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Create a mock TemplateError object
    class MockTemplateError:
        def __init__(self, message):
            self.message = message

    # Create an instance of UndefinedVariableInTemplate
    error = UndefinedVariableInTemplate(
        message="Test message",
        error=MockTemplateError("Test error message"),
        context={"key": "value"}
    )

    # Test the __str__ method
    assert str(error) == (
        "Test message. "
        "Error message: Test error message. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #28
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test with message
    exc_with_msg = CookiecutterException("Test error message")
    assert str(exc_with_msg) == "Test error message"

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #29
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the constructor of VCSNotInstalled exception."""
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #32
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_VCSNotInstalled():
    """Test VCSNotInstalled exception initialization."""
    exception = VCSNotInstalled("Git is not installed")
    assert str(exception) == "Git is not installed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #35
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #36
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    try:
        raise UnknownExtension("Test message")
    except UnknownExtension as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #37
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"


# LLM-generated content at query #38
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #39
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode JSON")
    assert str(exc) == "Failed to decode JSON"

    # Test with additional context
    exc_with_context = ContextDecodingException("Failed to decode JSON", "file.json")
    assert str(exc_with_context) == "Failed to decode JSON"

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #40
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #41
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #42
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #43
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException("Input directory is not templated")
    assert str(exception) == "Input directory is not templated"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #44
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #45
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Failed to decode JSON"


# LLM-generated content at query #46
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Project directory not found")
    assert str(exception) == "Project directory not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)

    # Test with message
    msg = "Test error message"
    exc_with_msg = CookiecutterException(msg)
    assert str(exc_with_msg) == msg

    # Test inheritance
    assert issubclass(CookiecutterException, Exception)


# LLM-generated content at query #49
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Test message"
    error = type('MockTemplateError', (), {'message': 'Test error message'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    result = str(exception)

    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Variable not defined"
    error = TemplateError("Template error occurred")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    result = str(exception)

    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #4
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Test message"
    error = TemplateError("Test error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    result = str(exception)
    expected = (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )

    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, InvalidConfiguration)
    assert str(exc) == ""

    # Test with a custom message
    custom_message = "Custom error message"
    exc_with_msg = InvalidConfiguration(custom_message)
    assert str(exc_with_msg) == custom_message


# LLM-generated content at query #6
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #7
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #8
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #9
#--------------------------

```python
def test_MissingProjectDir():
    """Test the MissingProjectDir exception."""
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #10
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_InvalidModeException():
    """Test that InvalidModeException is properly initialized."""
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #13
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    message = "Unknown extension"
    exception = UnknownExtension(message)
    assert str(exception) == message


# LLM-generated content at query #14
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Extension 'test_ext' could not be imported")
    assert str(exception) == "Extension 'test_ext' could not be imported"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #15
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #16
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #17
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #18
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration file")
    assert str(exception) == "Invalid configuration file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #19
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #20
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #21
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #22
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exc = FailedHookException("Hook failed")
    assert str(exc) == "Hook failed"

    # Test with additional context
    exc_with_context = FailedHookException("Hook failed", "pre_gen_project")
    assert str(exc_with_context) == "Hook failed"

    # Test that the exception is an instance of CookiecutterException
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #23
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exception = FailedHookException("Hook failed")
    assert str(exception) == "Hook failed"

    # Test with additional context
    exception_with_context = FailedHookException("Hook failed", "script.py", "pre_gen_project")
    assert str(exception_with_context) == "Hook failed. Hook script: script.py. Hook type: pre_gen_project"


# LLM-generated content at query #24
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #25
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #26
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #27
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #28
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #29
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #30
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the VCSNotInstalled exception constructor."""
    try:
        raise VCSNotInstalled("Git is not installed")
    except VCSNotInstalled as e:
        assert str(e) == "Git is not installed"
        assert isinstance(e, CookiecutterException)


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
    """Test the constructor of InvalidZipRepository."""
    try:
        raise InvalidZipRepository("Test message")
    except InvalidZipRepository as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #33
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #34
#--------------------------

```python
def test_UnknownRepoType():
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #35
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #36
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #37
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid config")
    assert str(exception) == "Invalid config"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #38
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #39
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #40
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #41
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON context")
    assert str(exception) == "Failed to decode JSON context"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #42
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = TemplateError("Undefined variable 'foo'")
    context = {'bar': 'baz'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is undefined. "
        "Error message: Undefined variable 'foo'. "
        f"Context: {context}"
    )


# LLM-generated content at query #43
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception constructor."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #44
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #45
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of UnknownExtension."""
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #46
#--------------------------

```python
def test_RepositoryNotFound():
    # Test default constructor
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    message = "Repository not found at given URL"
    exc_with_msg = RepositoryNotFound(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #47
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #48
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #49
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception constructor."""
    message = "Unknown extension error"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


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
def test_RepositoryNotFound():
    """Test that RepositoryNotFound exception is raised with the correct message."""
    message = "Repository not found"
    exc = RepositoryNotFound(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #52
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #53
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test that UnknownTemplateDirException can be instantiated."""
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #54
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Undefined variable in template"
    error = TemplateError("Variable not defined")
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


# LLM-generated content at query #55
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, ConfigDoesNotExistException)
    assert str(exception) == ""


# LLM-generated content at query #56
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #57
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Unknown repository type"


# LLM-generated content at query #58
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #59
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #60
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #61
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #62
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #63
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #64
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #65
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #66
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #67
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)

    # Test with a custom message
    custom_message = "Custom error message"
    exception_with_message = InvalidModeException(custom_message)
    assert str(exception_with_message) == custom_message

    # Test default message is empty
    assert str(exception) == ""


# LLM-generated content at query #68
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #69
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


# LLM-generated content at query #70
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test default constructor
    exception = RepositoryCloneFailed()
    assert str(exception) == ""

    # Test constructor with message
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message


# LLM-generated content at query #71
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #72
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #73
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #74
#--------------------------

```python
def test_InvalidZipRepository():
    # Test that InvalidZipRepository can be instantiated
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #75
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #76
#--------------------------

```python
def test_ContextDecodingException():
    """Test the constructor of ContextDecodingException."""
    message = "Failed to decode JSON context file"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #77
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test that RepositoryCloneFailed can be instantiated
    exception = RepositoryCloneFailed("Test message")
    assert isinstance(exception, RepositoryCloneFailed)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #78
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #79
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #80
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    # Test basic instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""

    # Test instantiation with a message
    message = "Repository not found at the specified URL"
    exception_with_message = RepositoryNotFound(message)
    assert str(exception_with_message) == message


# LLM-generated content at query #81
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #82
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #83
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #84
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exc = RepositoryNotFound(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #85
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #86
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception initialization."""
    exc = UnknownRepoType("Unknown repository type")
    assert str(exc) == "Unknown repository type"
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #87
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    hook_name = "pre_gen_project"
    exception = FailedHookException(message, hook_name)
    assert str(exception) == "Hook script failed: pre_gen_project"
    assert exception.message == message
    assert exception.hook_name == hook_name


# LLM-generated content at query #88
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #89
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception constructor."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #90
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, ConfigDoesNotExistException)
    assert str(exception) == ""


# LLM-generated content at query #91
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #92
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"

    # Test with no arguments
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""

    # Test that it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #93
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #94
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidModeException)
    assert str(exception) == ""


# LLM-generated content at query #95
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #96
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert isinstance(exception, RepositoryCloneFailed)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Failed to clone repository"


# LLM-generated content at query #97
#--------------------------

```python
def test_OutputDirExistsException():
    """Test the constructor of OutputDirExistsException."""
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #98
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"

    # Test with no message
    exc = ContextDecodingException()
    assert str(exc) == ""

    # Test that it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #99
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #100
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exc = FailedHookException("Hook failed")
    assert str(exc) == "Hook failed"

    # Test with additional context
    exc_with_context = FailedHookException("Hook failed", "pre_gen_project", {"key": "value"})
    assert str(exc_with_context) == "Hook failed. Hook: pre_gen_project. Context: {'key': 'value'}"


# LLM-generated content at query #101
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #102
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #103
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #104
#--------------------------

```python
def test_MissingProjectDir():
    """Test the MissingProjectDir exception constructor."""
    try:
        raise MissingProjectDir("Test message")
    except MissingProjectDir as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #105
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #106
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #107
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #108
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


# LLM-generated content at query #109
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #110
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #111
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #112
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #113
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #114
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidModeException)
    assert str(exception) == ""


# LLM-generated content at query #115
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #116
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #117
#--------------------------

```python
def test_UnknownRepoType():
    """Test the constructor of UnknownRepoType."""
    try:
        raise UnknownRepoType("This is a test message")
    except UnknownRepoType as e:
        assert str(e) == "This is a test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #118
#--------------------------

```python
def test_InvalidModeException():
    """Test the InvalidModeException constructor."""
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #119
#--------------------------

```python
def test_MissingProjectDir():
    """Test the MissingProjectDir exception constructor."""
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #120
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert exception.message == message


# LLM-generated content at query #121
#--------------------------

```python
def test_MissingProjectDir():
    """Test the MissingProjectDir exception constructor."""
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #122
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #123
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)

    # Test with a message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #124
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #125
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #126
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc_with_msg, Exception)


# LLM-generated content at query #127
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #128
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #129
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Test message"
    error = TemplateError("Test error")
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


# LLM-generated content at query #130
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception constructor."""
    exception = UnknownExtension("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #131
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the VCSNotInstalled exception constructor."""
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #132
#--------------------------

```python
def test_EmptyDirNameException():
    """Test the EmptyDirNameException constructor."""
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #133
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #134
#--------------------------

```python
def test_UnknownRepoType():
    """Test the constructor of UnknownRepoType exception."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #135
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #136
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #137
#--------------------------

```python
def test_InvalidConfiguration():
    # Test that InvalidConfiguration can be instantiated with a message
    message = "Invalid configuration provided"
    exception = InvalidConfiguration(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #138
#--------------------------

```python
def test_CookiecutterException():
    # Test default constructor
    exc = CookiecutterException()
    assert str(exc) == ""

    # Test constructor with message
    message = "Test exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message

    # Test inheritance from Exception
    assert isinstance(exc, Exception)


# LLM-generated content at query #139
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #140
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


# LLM-generated content at query #141
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #142
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #143
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration file")
    assert str(exception) == "Invalid configuration file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #144
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #145
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #146
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #147
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #148
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #149
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #150
#--------------------------

```python
def test_UnknownTemplateDirException():
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #151
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"


# LLM-generated content at query #152
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)


# LLM-generated content at query #153
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test message")
    except InvalidConfiguration as e:
        assert str(e) == "Test message"


# LLM-generated content at query #154
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Variable not defined"
    error = Mock()
    error.message = "Template error occurred"
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        "Variable not defined. "
        "Error message: Template error occurred. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #155
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #156
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #157
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #158
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #159
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #160
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = type('MockTemplateError', (), {'message': 'Template rendering error'})()
    context = {'bar': 'baz'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is undefined. "
        "Error message: Template rendering error. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #161
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #162
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #163
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #164
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #165
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #166
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #167
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #168
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #169
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #170
#--------------------------

```python
def test_CookiecutterException():
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)


# LLM-generated content at query #171
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


# LLM-generated content at query #172
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #173
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


# LLM-generated content at query #174
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #175
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation
    exc = InvalidModeException()
    assert isinstance(exc, InvalidModeException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test message can be set
    msg = "Custom error message"
    exc_with_msg = InvalidModeException(msg)
    assert str(exc_with_msg) == msg

    # Test default message when no args provided
    exc_default = InvalidModeException()
    assert str(exc_default) == ""


# LLM-generated content at query #176
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #177
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert exc.__class__.__bases__ == (CookiecutterException,)


# LLM-generated content at query #178
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #179
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #180
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #181
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #182
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #183
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation
    exc = RepositoryNotFound("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc_empty = RepositoryNotFound()
    assert str(exc_empty) == ""
    assert isinstance(exc_empty, CookiecutterException)


# LLM-generated content at query #184
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #185
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #186
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    try:
        raise CookiecutterException("Test exception message")
    except CookiecutterException as e:
        assert str(e) == "Test exception message"
        assert isinstance(e, Exception)


# LLM-generated content at query #187
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #188
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #189
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test the constructor of UnknownTemplateDirException."""
    exception = UnknownTemplateDirException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #190
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #191
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test that UnknownTemplateDirException can be instantiated."""
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #192
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #193
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    # Test basic instantiation
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "The repository is not a valid Zip archive"
    exc_with_msg = InvalidZipRepository(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #194
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the VCSNotInstalled exception constructor."""
    # Test default constructor
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test constructor with message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #195
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception constructor."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #196
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


# LLM-generated content at query #197
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = type('MockTemplateError', (), {'message': 'Template rendering error'})()
    context = {'bar': 'baz'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #198
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #199
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    try:
        raise UnknownExtension("Test message")
    except UnknownExtension as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #200
#--------------------------

```python
def test_OutputDirExistsException():
    """Test the constructor of OutputDirExistsException."""
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #201
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #202
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #203
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #204
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #205
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Generated project directory not found")
    assert str(exception) == "Generated project directory not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #206
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #207
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #208
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration")
    assert str(exception) == "Invalid configuration"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #209
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode JSON")
    assert str(exc) == "Failed to decode JSON"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc = ContextDecodingException()
    assert str(exc) == ""
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #210
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


# LLM-generated content at query #211
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test message")
    except InvalidConfiguration as e:
        assert str(e) == "Test message"


# LLM-generated content at query #212
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #213
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #214
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #215
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #216
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #217
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration"


# LLM-generated content at query #218
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #219
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #220
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of UnknownExtension."""
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #221
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #222
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #223
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #224
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of the UnknownExtension exception."""
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #225
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)

    # Test with message
    exc_with_msg = CookiecutterException("Test message")
    assert str(exc_with_msg) == "Test message"
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #226
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with additional context
    exc_with_context = ContextDecodingException("Another test", "extra context")
    assert str(exc_with_context) == "Another test"


# LLM-generated content at query #227
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON context file")
    assert str(exception) == "Failed to decode JSON context file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #228
#--------------------------

```python
def test_ConfigDoesNotExistException():
    """Test that ConfigDoesNotExistException can be instantiated."""
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #229
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #230
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #231
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    # Test basic instantiation
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    custom_msg = "Custom error message"
    exc_with_msg = RepositoryNotFound(custom_msg)
    assert str(exc_with_msg) == custom_msg


# LLM-generated content at query #232
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #233
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #234
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


# LLM-generated content at query #235
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #236
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #237
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)

    # Test with message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message
    assert exc_with_msg.args == (message,)

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_with_msg, CookiecutterException)


# LLM-generated content at query #238
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #239
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #240
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Undefined variable"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #241
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation
    exc = InvalidModeException()
    assert isinstance(exc, InvalidModeException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Test message"
    exc_with_msg = InvalidModeException(message)
    assert str(exc_with_msg) == message

    # Test default message
    exc_default = InvalidModeException()
    assert str(exc_default) == ""


# LLM-generated content at query #242
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    try:
        raise UnknownExtension("Test message")
    except UnknownExtension as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #243
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    hook_name = "pre_gen_project"
    exception = FailedHookException(message, hook_name)
    assert str(exception) == f"Hook script '{hook_name}' failed. {message}"
    assert exception.message == message
    assert exception.hook_name == hook_name


# LLM-generated content at query #244
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON")
    assert str(exception) == "Failed to decode JSON"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #245
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #246
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #247
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #248
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #249
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #250
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    hook_name = "pre_gen_project"
    exception = FailedHookException(message, hook_name)
    assert str(exception) == f"{message}: {hook_name}"
    assert exception.message == message
    assert exception.hook_name == hook_name


# LLM-generated content at query #251
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #252
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #253
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #254
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"


# LLM-generated content at query #255
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #256
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    custom_msg = "Custom error message"
    exc_with_msg = InvalidZipRepository(custom_msg)
    assert str(exc_with_msg) == custom_msg


# LLM-generated content at query #257
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #258
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"


# LLM-generated content at query #259
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #260
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #261
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #262
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #263
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #264
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #265
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #266
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Test message")
    except UnknownRepoType as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #267
#--------------------------

```python
def test_RepositoryNotFound():
    # Test default constructor
    exc = RepositoryNotFound()
    assert str(exc) == ""

    # Test constructor with custom message
    message = "Repository not found at the specified URL"
    exc = RepositoryNotFound(message)
    assert str(exc) == message


# LLM-generated content at query #268
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #269
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Test message"
    error = type('MockTemplateError', (), {'message': 'Test error message'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #270
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"bar": "baz"}
    exception = UndefinedVariableInTemplate(message, error, context)

    # Exercise
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #271
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test the constructor of NonTemplatedInputDirException."""
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, NonTemplatedInputDirException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #272
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #273
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"

    # Test with no arguments
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""

    # Test that it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #274
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #275
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #276
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable 'foo' is undefined"
    error = TemplateError("Error message")
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)

    # Verify
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context

    # Verify __str__ output
    expected_str = (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str


# LLM-generated content at query #277
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #278
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #279
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #280
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #281
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    """Test the __str__ method of UndefinedVariableInTemplate."""
    message = "Variable not defined"
    error = TemplateError("Error in template")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #282
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration provided")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration provided"


# LLM-generated content at query #283
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #284
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #285
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #286
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON context")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Failed to decode JSON context"


# LLM-generated content at query #287
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Undefined variable"
    error = TemplateError("Variable not defined")
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


# LLM-generated content at query #288
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = TemplateError("Error in template")
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


# LLM-generated content at query #289
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exc = FailedHookException()
    assert isinstance(exc, FailedHookException)
    assert isinstance(exc, CookiecutterException)

    # Test with message
    exc_with_msg = FailedHookException("Hook failed")
    assert str(exc_with_msg) == "Hook failed"

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #290
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #291
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)


# LLM-generated content at query #292
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #293
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Undefined variable in template"
    error = type('MockTemplateError', (), {'message': 'Variable not defined'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #294
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, MissingProjectDir)
    assert str(exception) == ""


# LLM-generated content at query #295
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception constructor."""
    message = "Unknown extension error"
    exception = UnknownExtension(message)

    assert isinstance(exception, UnknownExtension)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message


# LLM-generated content at query #296
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc_with_msg, Exception)


# LLM-generated content at query #297
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of UnknownExtension."""
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #298
#--------------------------

```python
def test_InvalidZipRepository():
    """Test that InvalidZipRepository can be instantiated."""
    try:
        raise InvalidZipRepository("Test message")
    except InvalidZipRepository as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #299
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #300
#--------------------------

```python
def test_MissingProjectDir():
    """Test the constructor of MissingProjectDir exception."""
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, MissingProjectDir)
    assert str(exception) == ""


# LLM-generated content at query #301
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test raise")
    except CookiecutterException as e:
        assert str(e) == "Test raise"
    except Exception:
        assert False, "Should have caught CookiecutterException"


# LLM-generated content at query #302
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #303
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


# LLM-generated content at query #304
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #305
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #306
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed("Failed to clone repository")
    assert str(exc) == "Failed to clone repository"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc_empty = RepositoryCloneFailed()
    assert str(exc_empty) == ""
    assert isinstance(exc_empty, CookiecutterException)


# LLM-generated content at query #307
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #308
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test that the exception can be instantiated with a message
    exc = RepositoryCloneFailed("Failed to clone repository")
    assert str(exc) == "Failed to clone repository"

    # Test that the exception is an instance of CookiecutterException
    assert isinstance(exc, CookiecutterException)

    # Test that the exception is an instance of Exception
    assert isinstance(exc, Exception)


# LLM-generated content at query #309
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Test error message"
    exc_with_msg = InvalidZipRepository(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #310
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Undefined variable in template"
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


# LLM-generated content at query #311
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #312
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #313
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, InvalidConfiguration)
    assert str(exc) == ""

    # Test with a custom message
    custom_message = "Custom error message"
    exc_with_msg = InvalidConfiguration(custom_message)
    assert str(exc_with_msg) == custom_message


# LLM-generated content at query #314
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    try:
        raise UnknownExtension("Test message")
    except UnknownExtension as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #315
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #316
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #317
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)


# LLM-generated content at query #318
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #319
#--------------------------

```python
def test_ContextDecodingException():
    """Test the ContextDecodingException constructor."""
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #320
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON")
    assert str(exception) == "Failed to decode JSON"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #321
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #322
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message


# LLM-generated content at query #323
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #324
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #325
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownRepoType)
    assert str(exception) == ""


# LLM-generated content at query #326
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #327
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #2
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode JSON")
    assert str(exc) == "Failed to decode JSON"

    # Test with additional context
    exc_with_context = ContextDecodingException("Invalid JSON format", "file.json")
    assert str(exc_with_context) == "Invalid JSON format"


# LLM-generated content at query #3
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #4
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #5
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Variable 'foo' is undefined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'bar': 'baz'}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        "Variable 'foo' is undefined. "
        "Error message: Template error occurred. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #6
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #7
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message


# LLM-generated content at query #8
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration provided")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration provided"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #11
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, InvalidConfiguration)
    assert str(exc) == ""

    # Test with a custom message
    custom_msg = "Custom error message"
    exc_with_msg = InvalidConfiguration(custom_msg)
    assert str(exc_with_msg) == custom_msg


# LLM-generated content at query #12
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #13
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #14
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #15
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #16
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #17
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Repository not found at the specified URL"
    exc_with_msg = RepositoryNotFound(message)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #18
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #19
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration file")
    assert str(exception) == "Invalid configuration file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #20
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #21
#--------------------------

```python
def test_InvalidZipRepository():
    # Test that InvalidZipRepository can be instantiated
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #22
#--------------------------

```python
def test_OutputDirExistsException():
    """Test the OutputDirExistsException constructor."""
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #23
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidConfiguration)
    assert str(exception) == ""


# LLM-generated content at query #24
#--------------------------

```python
def test_NonTemplatedInputDirException():
    # Test default constructor
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""

    # Test with custom message
    custom_message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(custom_message)
    assert str(exception) == custom_message


# LLM-generated content at query #25
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #26
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of UnknownExtension."""
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #27
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #28
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message


# LLM-generated content at query #29
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #30
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #31
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation
    exc = UnknownExtension()
    assert isinstance(exc, UnknownExtension)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with custom message
    custom_msg = "Custom error message"
    exc_with_msg = UnknownExtension(custom_msg)
    assert str(exc_with_msg) == custom_msg

    # Test default message is empty
    assert str(exc) == ""


# LLM-generated content at query #32
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the constructor of VCSNotInstalled exception."""
    exception = VCSNotInstalled("git is not installed")
    assert str(exception) == "git is not installed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #33
#--------------------------

```python
def test_MissingProjectDir():
    try:
        raise MissingProjectDir("Test message")
    except MissingProjectDir as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #34
#--------------------------

```python
def test_NonTemplatedInputDirException():
    """Test the constructor of NonTemplatedInputDirException."""
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, NonTemplatedInputDirException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #35
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #36
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #37
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation
    exc = UnknownExtension()
    assert isinstance(exc, UnknownExtension)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Test extension error"
    exc_with_msg = UnknownExtension(message)
    assert str(exc_with_msg) == message

    # Test default message when none provided
    exc_default = UnknownExtension()
    assert str(exc_default) == ""


# LLM-generated content at query #38
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation
    exc = UnknownExtension()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, UnknownExtension)
    assert str(exc) == ""

    # Test with custom message
    message = "Custom error message"
    exc = UnknownExtension(message)
    assert str(exc) == message

    # Test inheritance chain
    assert isinstance(exc, Exception)


# LLM-generated content at query #39
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #40
#--------------------------

```python
def test_MissingProjectDir():
    try:
        raise MissingProjectDir("Project directory not found")
    except MissingProjectDir as e:
        assert str(e) == "Project directory not found"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #41
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #42
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #43
#--------------------------

```python
def test_InvalidModeException():
    try:
        raise InvalidModeException("Test message")
    except InvalidModeException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #44
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #45
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"

    # Test with no message
    exc = ContextDecodingException()
    assert str(exc) == ""

    # Test that it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #46
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #49
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #50
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation
    exc = FailedHookException("Hook failed")
    assert str(exc) == "Hook failed"
    assert exc.args == ("Hook failed",)

    # Test with additional context
    exc_with_context = FailedHookException("Hook failed with context", "some_context")
    assert str(exc_with_context) == "Hook failed with context"
    assert exc_with_context.args == ("Hook failed with context", "some_context")

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #51
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #52
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Undefined variable in template"
    error = Exception("Template error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #53
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #54
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #55
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception constructor."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #56
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled("git is not installed")
    assert str(exception) == "git is not installed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #57
#--------------------------

```python
def test_UnknownRepoType():
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #58
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #59
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #60
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #61
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is not defined"
    error = object()  # Mock TemplateError
    error.message = "Template rendering error"
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is not defined. "
        "Error message: Template rendering error. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #62
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is undefined. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #63
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation
    exc = InvalidModeException()
    assert isinstance(exc, InvalidModeException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with custom message
    message = "Custom error message"
    exc_with_msg = InvalidModeException(message)
    assert str(exc_with_msg) == message

    # Test default message
    exc_default = InvalidModeException()
    assert str(exc_default) == ""

    # Test inheritance chain
    assert InvalidModeException.__bases__ == (CookiecutterException,)


# LLM-generated content at query #64
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #65
#--------------------------

```python
def test_VCSNotInstalled():
    vcs_error = VCSNotInstalled("git is not installed")
    assert str(vcs_error) == "git is not installed"


# LLM-generated content at query #66
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


# LLM-generated content at query #67
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


# LLM-generated content at query #68
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #69
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #70
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


# LLM-generated content at query #71
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #72
#--------------------------

```python
def test_UnknownRepoType():
    """Test the constructor of UnknownRepoType exception."""
    exception = UnknownRepoType("Unknown repository type")
    assert str(exception) == "Unknown repository type"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #73
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    # Test basic instantiation
    exc = InvalidConfiguration()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with a custom message
    custom_message = "Custom error message"
    exc_with_msg = InvalidConfiguration(custom_message)
    assert str(exc_with_msg) == custom_message


# LLM-generated content at query #74
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #75
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception constructor."""
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownRepoType)
    assert str(exception) == ""


# LLM-generated content at query #76
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #77
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


# LLM-generated content at query #78
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #79
#--------------------------

```python
def test_MissingProjectDir():
    """Test the constructor of MissingProjectDir."""
    try:
        raise MissingProjectDir("Test message")
    except MissingProjectDir as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #80
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #81
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #82
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed."""
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #83
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #84
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #85
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception class."""
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #86
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #87
#--------------------------

```python
def test_MissingProjectDir():
    # Test that MissingProjectDir can be instantiated
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)

    # Test that the exception can be instantiated with a custom message
    custom_message = "Custom error message"
    exception_with_message = MissingProjectDir(custom_message)
    assert str(exception_with_message) == custom_message


# LLM-generated content at query #88
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidZipRepository)
    assert str(exception) == ""


# LLM-generated content at query #89
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #90
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode JSON")
    assert str(exc) == "Failed to decode JSON"

    # Test with additional context
    exc_with_context = ContextDecodingException(
        "Failed to decode JSON",
        "The file contains invalid JSON syntax"
    )
    assert str(exc_with_context) == "Failed to decode JSON"

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #91
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test that it can be raised
    try:
        raise CookiecutterException("Test raise")
    except CookiecutterException as e:
        assert str(e) == "Test raise"
    except Exception:
        assert False, "Should have caught CookiecutterException"


# LLM-generated content at query #92
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed."""
    try:
        raise RepositoryCloneFailed("Test message")
    except RepositoryCloneFailed as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #93
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation
    exc = OutputDirExistsException()
    assert isinstance(exc, OutputDirExistsException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with a custom message
    custom_message = "Custom output directory exists message"
    exc_with_msg = OutputDirExistsException(custom_message)
    assert str(exc_with_msg) == custom_message

    # Test default message is empty string
    assert str(exc) == ""


# LLM-generated content at query #94
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #95
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #96
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Undefined variable"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    assert result == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #97
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #98
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #99
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #100
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, OutputDirExistsException)
    assert str(exception) == ""


# LLM-generated content at query #101
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #102
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #103
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #104
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Undefined variable"
    error = Exception("Template error")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #105
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #106
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with a custom message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #107
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #108
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #109
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #110
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #111
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #112
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #113
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)

    # Test with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test inheritance
    assert issubclass(CookiecutterException, Exception)


# LLM-generated content at query #114
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #115
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = Exception("Error message")
    context = {"bar": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #116
#--------------------------

```python
def test_InvalidModeException():
    """Test the InvalidModeException constructor."""
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #117
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #118
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #119
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is not defined"
    error = Mock()
    error.message = "Variable 'foo' is not defined in template"
    context = {"bar": "baz"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        "Variable 'foo' is not defined. "
        "Error message: Variable 'foo' is not defined in template. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #120
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #121
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the constructor of VCSNotInstalled exception."""
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #122
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)


# LLM-generated content at query #123
#--------------------------

```python
def test_InvalidConfiguration():
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #124
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

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #125
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #126
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #127
#--------------------------

```python
def test_UnknownExtension():
    """Test the UnknownExtension exception."""
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #128
#--------------------------

```python
def test_UnknownTemplateDirException():
    # Test that the exception can be instantiated
    exception = UnknownTemplateDirException()
    assert isinstance(exception, UnknownTemplateDirException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)

    # Test that the exception can be instantiated with a custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownTemplateDirException(custom_message)
    assert str(exception_with_message) == custom_message


# LLM-generated content at query #129
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #130
#--------------------------

```python
def test_UnknownRepoType():
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #131
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #132
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #133
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test default constructor
    exc = RepositoryCloneFailed()
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message


# LLM-generated content at query #134
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #135
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #136
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)


# LLM-generated content at query #137
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #138
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #139
#--------------------------

```python
def test_ContextDecodingException():
    """Test the constructor of ContextDecodingException."""
    message = "Failed to decode JSON context file"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #140
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #141
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #142
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Setup
    message = "Undefined variable in template"
    error = Exception("Template error occurred")
    context = {"key": "value"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)

    # Verify
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #143
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no arguments (should use default Exception behavior)
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""

    # Test that it's a subclass of CookiecutterException
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc_empty, CookiecutterException)


# LLM-generated content at query #144
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration file")
    assert str(exception) == "Invalid configuration file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #145
#--------------------------

```python
def test_MissingProjectDir():
    # Test that MissingProjectDir can be instantiated with no arguments
    exc = MissingProjectDir()
    assert isinstance(exc, MissingProjectDir)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""  # Default message should be empty

    # Test that MissingProjectDir can be instantiated with a custom message
    custom_message = "Custom error message"
    exc_with_msg = MissingProjectDir(custom_message)
    assert str(exc_with_msg) == custom_message


# LLM-generated content at query #146
#--------------------------

```python
def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #147
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration file")
    assert str(exception) == "Invalid configuration file"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #148
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #149
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #150
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Test message"
    error = Mock(spec=TemplateError)
    error.message = "Test error message"
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        "Test message. "
        "Error message: Test error message. "
        "Context: {'key': 'value'}"
    )


# LLM-generated content at query #151
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #152
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #153
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #154
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #155
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #156
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #157
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #158
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #159
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #160
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #161
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)
    assert str(exception) == ""


# LLM-generated content at query #162
#--------------------------

```python
def test_InvalidConfiguration():
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #163
#--------------------------

```python
def test_ContextDecodingException():
    """Test the constructor of ContextDecodingException."""
    message = "Failed to decode JSON context file"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #164
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #165
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with custom message
    custom_message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(custom_message)
    assert str(exc_with_msg) == custom_message

    # Test default message when no argument is provided
    exc_default = RepositoryCloneFailed()
    assert str(exc_default) == ""


# LLM-generated content at query #166
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #167
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #168
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #169
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #170
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #171
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #172
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


# LLM-generated content at query #173
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #174
#--------------------------

```python
def test_InvalidConfiguration():
    """Test the InvalidConfiguration exception."""
    exception = InvalidConfiguration("Invalid configuration provided")
    assert str(exception) == "Invalid configuration provided"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #175
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    # Test basic instantiation
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with a custom message
    custom_message = "Custom error message"
    exc_with_msg = RepositoryNotFound(custom_message)
    assert str(exc_with_msg) == custom_message


# LLM-generated content at query #176
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Project directory not found")
    assert str(exception) == "Project directory not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #177
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no message
    exc_no_msg = ContextDecodingException()
    assert str(exc_no_msg) == ""
    assert isinstance(exc_no_msg, CookiecutterException)


# LLM-generated content at query #178
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #179
#--------------------------

```python
def test_VCSNotInstalled():
    """Test that VCSNotInstalled exception can be instantiated."""
    try:
        raise VCSNotInstalled("Version control system not installed")
    except VCSNotInstalled as e:
        assert str(e) == "Version control system not installed"


# LLM-generated content at query #180
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test default constructor
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    message = "Failed to clone repository"
    exc_with_msg = RepositoryCloneFailed(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #181
#--------------------------

```python
def test_UnknownTemplateDirException():
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #182
#--------------------------

```python
def test_InvalidConfiguration():
    # Test that InvalidConfiguration can be instantiated with a message
    msg = "Invalid configuration file"
    exc = InvalidConfiguration(msg)
    assert str(exc) == msg
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #183
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


# LLM-generated content at query #184
#--------------------------

```python
def test_InvalidModeException():
    """Test the InvalidModeException constructor."""
    try:
        raise InvalidModeException("Test message")
    except InvalidModeException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #185
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Undefined variable"
    error = Mock()
    error.message = "Variable 'foo' is undefined"
    context = {"bar": "baz"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #186
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #187
#--------------------------

```python
def test_CookiecutterException():
    """Test the CookiecutterException constructor."""
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)


# LLM-generated content at query #188
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #189
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #190
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"


# LLM-generated content at query #191
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #192
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the VCSNotInstalled exception constructor."""
    # Test basic instantiation
    exc = VCSNotInstalled()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Git is not installed"
    exc_with_msg = VCSNotInstalled(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #193
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #194
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #195
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidModeException)
    assert str(exception) == ""


# LLM-generated content at query #196
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


# LLM-generated content at query #197
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test default constructor
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test with custom message
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert str(exc) == message


# LLM-generated content at query #198
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, UnknownTemplateDirException)


# LLM-generated content at query #199
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #200
#--------------------------

```python
def test_UnknownExtension():
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #201
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #202
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, NonTemplatedInputDirException)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #203
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = Exception("Error message")
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


# LLM-generated content at query #204
#--------------------------

```python
def test_UnknownRepoType():
    exception = UnknownRepoType()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #205
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #206
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration provided")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration provided"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #207
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Output directory already exists")
    assert str(exception) == "Output directory already exists"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #208
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException("Directory name cannot be empty")
    assert str(exception) == "Directory name cannot be empty"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #209
#--------------------------

```python
def test_CookiecutterException():
    # Test default constructor
    exc = CookiecutterException()
    assert str(exc) == ""

    # Test constructor with message
    message = "Test exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message


# LLM-generated content at query #210
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #211
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration provided")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration provided"


# LLM-generated content at query #212
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #213
#--------------------------

```python
def test_CookiecutterException():
    # Test default constructor
    exc = CookiecutterException()
    assert str(exc) == ""

    # Test constructor with message
    message = "Test exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message

    # Test inheritance
    assert isinstance(exc, Exception)


# LLM-generated content at query #214
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #215
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is undefined"
    error = MockTemplateError("Error message")
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

# Mock class for TemplateError
class MockTemplateError:
    def __init__(self, message):
        self.message = message


# LLM-generated content at query #216
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #217
#--------------------------

```python
def test_InvalidZipRepository():
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #218
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, CookiecutterException)

    # Test with no arguments (should work with default Exception behavior)
    exc_empty = ContextDecodingException()
    assert str(exc_empty) == ""

    # Test that it's a proper exception
    try:
        raise ContextDecodingException("Test error")
    except ContextDecodingException as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have caught ContextDecodingException"


# LLM-generated content at query #219
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


# LLM-generated content at query #220
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #221
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #222
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #223
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"


# LLM-generated content at query #224
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, ConfigDoesNotExistException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #225
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #226
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Test message"
    error = TemplateError("Test error message")
    context = {"key": "value"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #227
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #228
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test that RepositoryCloneFailed is a subclass of CookiecutterException
    assert issubclass(RepositoryCloneFailed, CookiecutterException)

    # Test that RepositoryCloneFailed can be instantiated with a message
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert str(exception) == message
    assert exception.args == (message,)

    # Test that RepositoryCloneFailed can be instantiated without a message
    exception_no_msg = RepositoryCloneFailed()
    assert str(exception_no_msg) == ""
    assert exception_no_msg.args == ()


# LLM-generated content at query #229
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #230
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #231
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #232
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException("Failed to decode JSON context")
    assert str(exception) == "Failed to decode JSON context"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #233
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #234
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode context")
    assert str(exc) == "Failed to decode context"

    # Test with additional arguments
    exc_with_args = ContextDecodingException("Failed to decode context", "file.json", "utf-8")
    assert str(exc_with_args) == "Failed to decode context"

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #235
#--------------------------

```python
def test_InvalidZipRepository():
    # Test default constructor
    exc = InvalidZipRepository()
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""

    # Test constructor with message
    msg = "Test message"
    exc = InvalidZipRepository(msg)
    assert str(exc) == msg


# LLM-generated content at query #236
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exc = ContextDecodingException("Failed to decode context")
    assert str(exc) == "Failed to decode context"

    # Test with additional context
    exc_with_context = ContextDecodingException("Failed to decode context", "file.json")
    assert str(exc_with_context) == "Failed to decode context"

    # Test inheritance
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)


# LLM-generated content at query #237
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #238
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    message = "Undefined variable in template"
    error = type('MockTemplateError', (), {'message': 'Variable not found'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #239
#--------------------------

```python
def test_InvalidModeException():
    """Test the constructor of InvalidModeException."""
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidModeException)
    assert str(exception) == ""


# LLM-generated content at query #240
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #241
#--------------------------

```python
def test_CookiecutterException():
    """Test the constructor of CookiecutterException."""
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message

    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test raise")
    except CookiecutterException as e:
        assert str(e) == "Test raise"
    except Exception:
        assert False, "Should have caught CookiecutterException"


# LLM-generated content at query #242
#--------------------------

```python
def test_UnknownRepoType():
    try:
        raise UnknownRepoType("Test message")
    except UnknownRepoType as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #243
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #244
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)

    # Test with no message
    exception = ContextDecodingException()
    assert str(exception) == ""
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #245
#--------------------------

```python
def test_FailedHookException():
    """Test the constructor of FailedHookException."""
    message = "Hook failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #246
#--------------------------

```python
def test_UnknownTemplateDirException():
    """Test the constructor of UnknownTemplateDirException."""
    try:
        raise UnknownTemplateDirException("Test message")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #247
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #248
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable 'foo' is undefined"
    error = Exception("Error message")
    context = {"bar": "baz"}

    exc = UndefinedVariableInTemplate(message, error, context)

    assert exc.message == message
    assert exc.error == error
    assert exc.context == context
    assert str(exc) == (
        f"{message}. "
        f"Error message: {error}. "
        f"Context: {context}"
    )


# LLM-generated content at query #249
#--------------------------

```python
def test_UnknownRepoType():
    # Test that UnknownRepoType can be instantiated with a message
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #250
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #251
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #252
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Setup
    message = "Test message"
    error = Exception("Test error")
    context = {"key": "value"}

    # Execution
    exception = UndefinedVariableInTemplate(message, error, context)

    # Assertions
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #253
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #254
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test error message")
    except InvalidConfiguration as e:
        assert str(e) == "Test error message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #255
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


# LLM-generated content at query #256
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #257
#--------------------------

```python
def test_RepositoryCloneFailed():
    """Test the constructor of RepositoryCloneFailed."""
    # Test basic instantiation
    exception = RepositoryCloneFailed("Failed to clone repository")
    assert str(exception) == "Failed to clone repository"
    assert isinstance(exception, CookiecutterException)

    # Test with no message
    exception = RepositoryCloneFailed()
    assert str(exception) == ""
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #258
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message


# LLM-generated content at query #259
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    message = "Variable not defined"
    error = type('MockTemplateError', (), {'message': 'Template error occurred'})()
    context = {'key': 'value'}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #260
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #261
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #262
#--------------------------

```python
def test_MissingProjectDir():
    exception = MissingProjectDir("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #263
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of the UnknownExtension exception."""
    message = "Unknown extension error"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #264
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the InvalidZipRepository exception constructor."""
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #265
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #266
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #267
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == "Test message"


# LLM-generated content at query #268
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType can be instantiated."""
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #269
#--------------------------

```python
def test_VCSNotInstalled():
    """Test the VCSNotInstalled exception constructor."""
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #270
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #271
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #272
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test with message
    message = "Test exception message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message
    assert exc_with_msg.args == (message,)

    # Test without message
    exc_no_msg = CookiecutterException()
    assert str(exc_no_msg) == ""
    assert exc_no_msg.args == ()


# LLM-generated content at query #273
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable not defined"
    error = Exception("Template error")
    context = {"key": "value"}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)

    # Verify
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #274
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, NonTemplatedInputDirException)
    assert str(exception) == ""


# LLM-generated content at query #275
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #276
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Repository not found")
    assert str(exception) == "Repository not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #277
#--------------------------

```python
def test_UnknownRepoType():
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #278
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


# LLM-generated content at query #279
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Test message")
    except InvalidConfiguration as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #280
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #281
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #282
#--------------------------

```python
def test_UnknownExtension():
    """Test the constructor of UnknownExtension."""
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #283
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidZipRepository)
    assert str(exception) == ""


# LLM-generated content at query #284
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #285
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #286
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #287
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #288
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #289
#--------------------------

```python
def test_UnknownRepoType():
    """Test the UnknownRepoType exception."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #290
#--------------------------

```python
def test_CookiecutterException():
    """Test the constructor of CookiecutterException."""
    # Test default constructor
    exc = CookiecutterException()
    assert str(exc) == ""

    # Test constructor with message
    message = "Test exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message

    # Test inheritance from Exception
    assert isinstance(exc, Exception)


# LLM-generated content at query #291
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #292
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #293
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #294
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, VCSNotInstalled)
    assert str(exception) == ""


# LLM-generated content at query #295
#--------------------------

```python
def test_InvalidConfiguration():
    """Test that InvalidConfiguration can be instantiated."""
    try:
        raise InvalidConfiguration("Invalid configuration")
    except InvalidConfiguration as e:
        assert str(e) == "Invalid configuration"


# LLM-generated content at query #296
#--------------------------

```python
def test_EmptyDirNameException():
    try:
        raise EmptyDirNameException("Directory name cannot be empty")
    except EmptyDirNameException as e:
        assert str(e) == "Directory name cannot be empty"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #297
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    hook_name = "pre_gen_project"
    exception = FailedHookException(message, hook_name)
    assert str(exception) == f"Hook script '{hook_name}' failed. {message}"


# LLM-generated content at query #298
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


# LLM-generated content at query #299
#--------------------------

```python
def test_InvalidConfiguration():
    # Test that InvalidConfiguration is a subclass of CookiecutterException
    assert issubclass(InvalidConfiguration, CookiecutterException)

    # Test that InvalidConfiguration can be instantiated with a message
    msg = "Invalid configuration file"
    exc = InvalidConfiguration(msg)
    assert str(exc) == msg
    assert isinstance(exc, CookiecutterException)

    # Test that InvalidConfiguration can be instantiated without a message
    exc_no_msg = InvalidConfiguration()
    assert str(exc_no_msg) == ""
    assert isinstance(exc_no_msg, CookiecutterException)


# LLM-generated content at query #300
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #301
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #302
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)
    assert str(exception) == ""


# LLM-generated content at query #303
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation
    exc = CookiecutterException("Test message")
    assert str(exc) == "Test message"

    # Test default message
    exc_default = CookiecutterException()
    assert str(exc_default) == ""

    # Test inheritance from Exception
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)


# LLM-generated content at query #304
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #305
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


# LLM-generated content at query #306
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    exception = InvalidZipRepository("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #307
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, EmptyDirNameException)


# LLM-generated content at query #308
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert exc.args == (message,)


# LLM-generated content at query #309
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #310
#--------------------------

```python
def test_RepositoryNotFound():
    """Test the constructor of RepositoryNotFound exception."""
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #311
#--------------------------

```python
def test_FailedHookException():
    message = "Hook script failed"
    exception = FailedHookException(message)
    assert str(exception) == message


# LLM-generated content at query #312
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #313
#--------------------------

```python
def test_InvalidZipRepository():
    """Test the constructor of InvalidZipRepository."""
    try:
        raise InvalidZipRepository("Test message")
    except InvalidZipRepository as e:
        assert str(e) == "Test message"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #314
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
        f"{message}. Error message: {error.message}. Context: {context}"
    )


# LLM-generated content at query #315
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable 'foo' is not defined"
    error = type('MockTemplateError', (), {'message': 'Template rendering failed'})()
    context = {'bar': 'baz'}

    # Exercise
    exception = UndefinedVariableInTemplate(message, error, context)
    result = str(exception)

    # Verify
    assert result == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #316
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #317
#--------------------------

```python
def test_UnknownRepoType():
    with pytest.raises(UnknownRepoType):
        raise UnknownRepoType()


# LLM-generated content at query #318
#--------------------------

```python
def test_UnknownRepoType():
    """Test that UnknownRepoType exception can be instantiated."""
    try:
        raise UnknownRepoType("Unknown repository type")
    except UnknownRepoType as e:
        assert str(e) == "Unknown repository type"
        assert isinstance(e, CookiecutterException)


# LLM-generated content at query #319
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #320
#--------------------------

```python
def test_OutputDirExistsException():
    exception = OutputDirExistsException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #321
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, InvalidModeException)
    assert str(exception) == ""


# LLM-generated content at query #322
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""


# LLM-generated content at query #323
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #324
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""


# LLM-generated content at query #325
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)


