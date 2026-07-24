####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test with simple values
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project"}
    exception = UndefinedVariableInTemplate(
        "Template variable error", error, context
    )
    
    expected_str = (
        "Template variable error. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "Undefined variable", error, context
    )
    
    expected_str = (
        "Undefined variable. "
        "Error message: Missing variable. "
        "Context: {}"
    )
    assert str(exception) == expected_str
    
    # Test with complex context
    error = MockTemplateError("'bar' not found")
    context = {"name": "test", "version": 1.0, "options": ["a", "b"]}
    exception = UndefinedVariableInTemplate(
        "Rendering failed", error, context
    )
    
    expected_str = (
        "Rendering failed. "
        "Error message: 'bar' not found. "
        "Context: {'name': 'test', 'version': 1.0, 'options': ['a', 'b']}"
    )
    assert str(exception) == expected_str
    
    # Test with real UndefinedError from jinja2
    try:
        # This would normally raise UndefinedError
        pass
    except:
        pass
    
    # Verify the exception attributes are properly set
    error = MockTemplateError("Test error")
    context = {"key": "value"}
    exception = UndefinedVariableInTemplate("Test", error, context)
    
    assert exception.message == "Test"
    assert exception.error == error
    assert exception.context == context


# LLM-generated content at query #2
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Test message"
    exception_with_msg = NonTemplatedInputDirException(message)
    assert str(exception_with_msg) == message
    assert isinstance(exception_with_msg, CookiecutterException)


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Invalid zip file format"
    exception_with_message = InvalidZipRepository(custom_message)
    assert str(exception_with_message) == custom_message
    assert isinstance(exception_with_message, InvalidZipRepository)
    
    # Test inheritance chain
    assert issubclass(InvalidZipRepository, CookiecutterException)
    assert issubclass(InvalidZipRepository, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidZipRepository("Test error")
    except InvalidZipRepository as e:
        assert str(e) == "Test error"
    
    # Test multiple inheritance levels
    exception = InvalidZipRepository()
    assert type(exception).__name__ == "InvalidZipRepository"
    assert exception.__class__.__name__ == "InvalidZipRepository"


# LLM-generated content at query #4
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test case 1: Basic functionality
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "author": "John Doe"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'author': 'John Doe'}"
    )
    assert str(exception) == expected_str
    
    # Test case 2: Empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "No variables defined",
        error,
        context
    )
    
    expected_str = "No variables defined. Error message: Missing variable. Context: {}"
    assert str(exception) == expected_str
    
    # Test case 3: Complex context with nested structures
    error = MockTemplateError("'config.database.host' not found")
    context = {
        "project": {
            "name": "MyApp",
            "version": "1.0"
        },
        "settings": ["debug", "production"]
    }
    exception = UndefinedVariableInTemplate(
        "Configuration error in template",
        error,
        context
    )
    
    expected_str = (
        "Configuration error in template. "
        "Error message: 'config.database.host' not found. "
        "Context: {'project': {'name': 'MyApp', 'version': '1.0'}, 'settings': ['debug', 'production']}"
    )
    assert str(exception) == expected_str
    
    # Test case 4: Special characters in messages
    error = MockTemplateError("Variable 'user-name' contains invalid characters")
    context = {"user": "test@example.com"}
    exception = UndefinedVariableInTemplate(
        "Validation failed: user-name",
        error,
        context
    )
    
    expected_str = (
        "Validation failed: user-name. "
        "Error message: Variable 'user-name' contains invalid characters. "
        "Context: {'user': 'test@example.com'}"
    )
    assert str(exception) == expected_str
    
    # Test case 5: Verify the exception stores the attributes correctly
    error = MockTemplateError("Test error")
    context = {"test": "value"}
    exception = UndefinedVariableInTemplate("Test message", error, context)
    
    assert exception.message == "Test message"
    assert exception.error == error
    assert exception.error.message == "Test error"
    assert exception.context == {"test": "value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic instantiation with no arguments
    exc = InvalidConfiguration()
    assert isinstance(exc, InvalidConfiguration)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Invalid configuration file format"
    exc = InvalidConfiguration(message)
    assert str(exc) == message

    # Test instantiation with a message and custom attributes
    exc = InvalidConfiguration("Custom error")
    exc.custom_attr = "test"
    assert exc.custom_attr == "test"

    # Test exception inheritance chain
    assert InvalidConfiguration.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)

    # Test that it can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by InvalidConfiguration"

    # Test with empty string message
    exc = InvalidConfiguration("")
    assert str(exc) == ""

    # Test with special characters in message
    special_msg = "Error: Invalid config @ line 5"
    exc = InvalidConfiguration(special_msg)
    assert str(exc) == special_msg


# LLM-generated content at query #6
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    custom_message = "Directory name cannot be empty"
    exception_with_message = EmptyDirNameException(custom_message)
    assert str(exception_with_message) == custom_message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #7
#--------------------------

def test_UndefinedVariableInTemplate():
    from jinja2 import UndefinedError
    error = UndefinedError("'foo' is undefined")
    context = {"project_name": "test_project"}
    exception = UndefinedVariableInTemplate(
        "Variable not defined in template",
        error,
        context
    )
    assert exception.message == "Variable not defined in template"
    assert exception.error == error
    assert exception.context == context
    expected_str = (
        "Variable not defined in template. "
        "Error message: 'foo' is undefined. "
        "Context: {'project_name': 'test_project'}"
    )
    assert str(exception) == expected_str


# LLM-generated content at query #8
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic instantiation with no arguments
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Config file not found at /path/to/config.yaml"
    exc_with_msg = ConfigDoesNotExistException(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    try:
        raise ConfigDoesNotExistException("Test error")
    except ConfigDoesNotExistException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert ConfigDoesNotExistException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #9
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation with default message
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Test custom message"
    exception_with_message = InvalidModeException(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise InvalidModeException("Test raising")
    except InvalidModeException as e:
        assert str(e) == "Test raising"
    
    # Test inheritance chain
    assert InvalidModeException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #10
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #11
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test basic initialization
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "author": "Test Author"}
    exception = UndefinedVariableInTemplate(
        "Template variable error", error, context
    )
    
    assert exception.message == "Template variable error"
    assert exception.error == error
    assert exception.context == context
    
    # Test string representation
    expected_str = (
        "Template variable error. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    error2 = MockTemplateError("Missing variable")
    exception2 = UndefinedVariableInTemplate(
        "Another error", error2, {}
    )
    assert str(exception2) == "Another error. Error message: Missing variable. Context: {}"
    
    # Test with complex error message
    error3 = MockTemplateError("'undefined' is undefined")
    context3 = {"config": {"key": "value"}, "list": [1, 2, 3]}
    exception3 = UndefinedVariableInTemplate(
        "Complex template error", error3, context3
    )
    assert "Complex template error" in str(exception3)
    assert "'undefined' is undefined" in str(exception3)
    assert "{'config': {'key': 'value'}, 'list': [1, 2, 3]}" in str(exception3)


# LLM-generated content at query #12
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #13
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic instantiation with no arguments
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Config file not found at /path/to/config.yaml"
    exc_with_msg = ConfigDoesNotExistException(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message
    
    # Test inheritance chain
    assert ConfigDoesNotExistException.__doc__ is not None
    assert "Exception for missing config file." in ConfigDoesNotExistException.__doc__
    
    # Test multiple arguments
    exc_multi = ConfigDoesNotExistException("Error", "additional", "info")
    assert exc_multi.args == ("Error", "additional", "info")


# LLM-generated content at query #14
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic instantiation with no arguments
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Config file not found at /path/to/config.yaml"
    exc_with_msg = ConfigDoesNotExistException(message)
    assert str(exc_with_msg) == message
    
    # Test that it can be raised and caught
    try:
        raise ConfigDoesNotExistException("Test error")
    except ConfigDoesNotExistException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert ConfigDoesNotExistException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #15
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc1 = InvalidZipRepository()
    assert isinstance(exc1, InvalidZipRepository)
    assert isinstance(exc1, CookiecutterException)
    assert str(exc1) == ""
    
    # Test instantiation with a message
    message = "Invalid zip file format"
    exc2 = InvalidZipRepository(message)
    assert exc2.args[0] == message
    assert str(exc2) == message
    
    # Test instantiation with multiple arguments
    exc3 = InvalidZipRepository("Error", "Additional info")
    assert exc3.args == ("Error", "Additional info")
    assert str(exc3) == "('Error', 'Additional info')"
    
    # Test inheritance chain
    assert issubclass(InvalidZipRepository, CookiecutterException)
    assert issubclass(InvalidZipRepository, Exception)


# LLM-generated content at query #16
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Test error message"
    exception_with_message = NonTemplatedInputDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #17
#--------------------------

```python
def test_UnknownTemplateDirException():
    # Test basic instantiation
    exception = UnknownTemplateDirException()
    assert isinstance(exception, UnknownTemplateDirException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownTemplateDirException(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that exception can be raised and caught
    try:
        raise UnknownTemplateDirException("Test error")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert UnknownTemplateDirException.__doc__ is not None
    assert "Exception for ambiguous project template directory" in UnknownTemplateDirException.__doc__


# LLM-generated content at query #18
#--------------------------

```python
def test_InvalidZipRepository():
    # Test that InvalidZipRepository can be instantiated without arguments
    exc1 = InvalidZipRepository()
    assert isinstance(exc1, InvalidZipRepository)
    assert isinstance(exc1, CookiecutterException)
    assert str(exc1) == ""
    
    # Test that InvalidZipRepository can be instantiated with a message
    message = "Invalid zip repository format"
    exc2 = InvalidZipRepository(message)
    assert exc2.args[0] == message
    assert str(exc2) == message
    
    # Test that InvalidZipRepository can be instantiated with multiple arguments
    exc3 = InvalidZipRepository("Error", "Additional info")
    assert exc3.args == ("Error", "Additional info")
    assert str(exc3) == "('Error', 'Additional info')"
    
    # Test inheritance chain
    assert issubclass(InvalidZipRepository, CookiecutterException)
    assert issubclass(InvalidZipRepository, Exception)


# LLM-generated content at query #19
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Input directory should be templated"
    exception_with_message = NonTemplatedInputDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #20
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Invalid zip file format"
    exc = InvalidZipRepository(message)
    assert exc.args == (message,)
    assert str(exc) == message
    
    # Test instantiation with multiple arguments
    exc = InvalidZipRepository("Invalid zip", "Corrupted archive")
    assert exc.args == ("Invalid zip", "Corrupted archive")
    assert str(exc) == "('Invalid zip', 'Corrupted archive')"
    
    # Test inheritance chain
    assert InvalidZipRepository.__doc__ is not None
    assert "Exception for bad zip repo" in InvalidZipRepository.__doc__


# LLM-generated content at query #21
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with message
    exc = UnknownExtension("Test extension error")
    assert str(exc) == "Test extension error"
    assert isinstance(exc, CookiecutterException)
    
    # Test instantiation without message
    exc2 = UnknownExtension()
    assert str(exc2) == ""
    
    # Test inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)
    
    # Test exception can be raised and caught
    try:
        raise UnknownExtension("Custom error message")
    except UnknownExtension as e:
        assert str(e) == "Custom error message"
    
    # Test with empty string message
    exc3 = UnknownExtension("")
    assert str(exc3) == ""
    
    # Test that it can be caught as base exception
    try:
        raise UnknownExtension("Test")
    except CookiecutterException:
        assert True
    else:
        assert False


# LLM-generated content at query #22
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert RepositoryNotFound.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)
    
    # Test exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by RepositoryNotFound"
    
    # Test multiple inheritance levels
    try:
        raise RepositoryNotFound("Another test")
    except CookiecutterException:
        assert True
    except Exception:
        assert False, "Should have been caught by CookiecutterException"


# LLM-generated content at query #23
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic exception instantiation
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository type could not be determined"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert UnknownRepoType.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)
    
    # Test exception can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have caught UnknownRepoType"
    
    # Test multiple arguments
    exception_multi = UnknownRepoType("Error", "additional", "args")
    assert exception_multi.args == ("Error", "additional", "args")


# LLM-generated content at query #24
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation with no arguments
    exception = CookiecutterException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Test error message"
    exception_with_msg = CookiecutterException(message)
    assert str(exception_with_msg) == message
    
    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test raising")
    except CookiecutterException as e:
        assert str(e) == "Test raising"
    
    # Test inheritance chain
    assert CookiecutterException.__base__ == Exception
    assert CookiecutterException.__name__ == "CookiecutterException"


# LLM-generated content at query #25
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Invalid zip file format"
    exc_with_msg = InvalidZipRepository(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message
    
    # Test instantiation with multiple arguments
    multi_args = InvalidZipRepository("Error", "Additional info")
    assert multi_args.args == ("Error", "Additional info")
    assert str(multi_args) == "('Error', 'Additional info')"
    
    # Test inheritance chain
    assert InvalidZipRepository.__doc__ is not None
    assert "bad zip repo" in InvalidZipRepository.__doc__.lower()


# LLM-generated content at query #26
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Repository not found at specified location"
    exc = RepositoryNotFound(custom_msg)
    assert str(exc) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert RepositoryNotFound.__mro__[0] is RepositoryNotFound
    assert RepositoryNotFound.__mro__[1] is CookiecutterException
    assert RepositoryNotFound.__mro__[2] is Exception
    assert RepositoryNotFound.__mro__[3] is BaseException
    assert RepositoryNotFound.__mro__[4] is object


# LLM-generated content at query #27
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with a message
    message = "Test extension error"
    exc = UnknownExtension(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test instantiation without a message
    exc2 = UnknownExtension()
    assert str(exc2) == ""

    # Test that it can be raised and caught
    try:
        raise UnknownExtension("Custom error message")
    except UnknownExtension as e:
        assert str(e) == "Custom error message"
    except Exception:
        pytest.fail("Should have been caught as UnknownExtension")

    # Test inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)


# LLM-generated content at query #28
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Invalid zip file format"
    exc_with_msg = InvalidZipRepository(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert InvalidZipRepository.__mro__[0] is InvalidZipRepository
    assert InvalidZipRepository.__mro__[1] is CookiecutterException
    assert InvalidZipRepository.__mro__[2] is Exception
    assert InvalidZipRepository.__mro__[3] is BaseException
    assert InvalidZipRepository.__mro__[4] is object


# LLM-generated content at query #29
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic exception instantiation
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Project directory 'my_project' not found"
    exception_with_msg = MissingProjectDir(custom_message)
    assert str(exception_with_msg) == custom_message
    
    # Test exception inheritance chain
    assert MissingProjectDir.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)
    
    # Test exception can be raised and caught
    try:
        raise MissingProjectDir("Test error")
    except MissingProjectDir as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by MissingProjectDir"
    
    # Test multiple arguments
    exception_multi = MissingProjectDir("Error", "additional", "args")
    assert str(exception_multi) == "('Error', 'additional', 'args')"


# LLM-generated content at query #30
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic instantiation with no arguments
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Config file not found at /path/to/config.yaml"
    exc = ConfigDoesNotExistException(message)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    try:
        raise ConfigDoesNotExistException("Test error")
    except ConfigDoesNotExistException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert ConfigDoesNotExistException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #31
#--------------------------

```python
def test_UnknownRepoType():
    # Test that UnknownRepoType is a subclass of CookiecutterException
    assert issubclass(UnknownRepoType, CookiecutterException)
    
    # Test instantiation with default message
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Custom error message"
    exception = UnknownRepoType(custom_message)
    assert str(exception) == custom_message
    
    # Test that exception can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    try:
        raise UnknownRepoType()
    except CookiecutterException:
        assert True
    else:
        assert False


# LLM-generated content at query #32
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic exception instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Output directory already exists at /path/to/dir"
    exception_with_msg = OutputDirExistsException(custom_message)
    assert str(exception_with_msg) == custom_message
    
    # Test exception inheritance chain
    assert exception.__class__.__name__ == "OutputDirExistsException"
    assert exception.__class__.__bases__[0].__name__ == "CookiecutterException"
    
    # Test exception can be raised and caught
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
    except CookiecutterException:
        pass  # Should also be catchable as parent type
    except Exception:
        pass  # Should also be catchable as base Exception type


# LLM-generated content at query #33
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Invalid zip repository: corrupted archive"
    exc_with_msg = InvalidZipRepository(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message

    # Test inheritance chain
    assert InvalidZipRepository.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)

    # Test multiple arguments
    multi_arg_exc = InvalidZipRepository("Error", "additional", "info")
    assert multi_arg_exc.args == ("Error", "additional", "info")
    assert str(multi_arg_exc) == "('Error', 'additional', 'info')"


# LLM-generated content at query #34
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation with no arguments
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message
    
    # Test instantiation with multiple arguments
    exc_multi = CookiecutterException("Error", "code", 500)
    assert str(exc_multi) == "('Error', 'code', 500)"
    
    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test exception")
    except CookiecutterException as e:
        assert str(e) == "Test exception"
    
    # Test inheritance chain
    assert CookiecutterException.__base__ == Exception
    assert CookiecutterException.__name__ == "CookiecutterException"


# LLM-generated content at query #35
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic exception instantiation
    exc = RepositoryCloneFailed("Failed to clone repository")
    assert str(exc) == "Failed to clone repository"
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)

    # Test exception with custom message
    custom_msg = "Cannot clone from https://example.com/template.git"
    exc = RepositoryCloneFailed(custom_msg)
    assert str(exc) == custom_msg

    # Test exception inheritance chain
    assert issubclass(RepositoryCloneFailed, CookiecutterException)
    assert issubclass(RepositoryCloneFailed, Exception)

    # Test exception can be raised and caught
    try:
        raise RepositoryCloneFailed("Test error")
    except RepositoryCloneFailed as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by RepositoryCloneFailed"

    # Test exception with empty message
    exc = RepositoryCloneFailed("")
    assert str(exc) == ""

    # Test exception with special characters in message
    special_msg = "Error: Cannot clone repo with name 'my-template@v1.0'"
    exc = RepositoryCloneFailed(special_msg)
    assert str(exc) == special_msg


# LLM-generated content at query #36
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Directory name cannot be empty"
    exception_with_msg = EmptyDirNameException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    
    # Test exception can be raised and caught
    try:
        raise EmptyDirNameException("Test error")
    except EmptyDirNameException as e:
        assert str(e) == "Test error"
    
    # Test multiple inheritance catching
    try:
        raise EmptyDirNameException()
    except CookiecutterException:
        assert True
    except Exception:
        assert True


# LLM-generated content at query #37
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #38
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #39
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException("Config file not found")
    assert str(exception) == "Config file not found"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #40
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic instantiation
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance hierarchy
    assert UnknownRepoType.__doc__ is not None
    assert "Exception for unknown repo types." in UnknownRepoType.__doc__
    assert "Raised if a repo's type cannot be determined." in UnknownRepoType.__doc__
    
    # Test multiple arguments
    multi_arg_exception = UnknownRepoType("Error", "additional", "args")
    assert str(multi_arg_exception) == "('Error', 'additional', 'args')"


# LLM-generated content at query #41
#--------------------------

```python
def test_CookiecutterException():
    # Test basic exception instantiation with no arguments
    exc = CookiecutterException()
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test exception with a message
    message = "Test error message"
    exc_with_msg = CookiecutterException(message)
    assert str(exc_with_msg) == message
    
    # Test exception inheritance chain
    assert CookiecutterException.__bases__ == (Exception,)
    
    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test raising")
    except CookiecutterException as e:
        assert str(e) == "Test raising"
    
    # Test that other Cookiecutter exceptions inherit from it
    assert issubclass(NonTemplatedInputDirException, CookiecutterException)
    assert issubclass(ConfigDoesNotExistException, CookiecutterException)
    assert issubclass(UndefinedVariableInTemplate, CookiecutterException)


# LLM-generated content at query #42
#--------------------------

```python
def test_VCSNotInstalled():
    # Test basic instantiation with no arguments
    exception = VCSNotInstalled()
    assert isinstance(exception, VCSNotInstalled)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Git is not installed on this system"
    exception_with_msg = VCSNotInstalled(custom_message)
    assert str(exception_with_msg) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise VCSNotInstalled("Test error")
    except VCSNotInstalled as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert VCSNotInstalled.__doc__ is not None
    assert "Exception when version control is unavailable" in VCSNotInstalled.__doc__


# LLM-generated content at query #43
#--------------------------

```python
def test_CookiecutterException():
    # Test basic exception instantiation
    exc = CookiecutterException("Test message")
    assert str(exc) == "Test message"
    assert isinstance(exc, Exception)
    
    # Test exception with no message
    exc = CookiecutterException()
    assert str(exc) == ""
    
    # Test exception inheritance
    assert issubclass(CookiecutterException, Exception)
    
    # Test exception with custom attributes
    exc = CookiecutterException("Error occurred")
    exc.custom_attr = "value"
    assert exc.custom_attr == "value"


# LLM-generated content at query #44
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with message
    exception = UnknownExtension("Test extension error")
    assert str(exception) == "Test extension error"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)

    # Test instantiation without message
    exception = UnknownExtension()
    assert str(exception) == ""

    # Test exception inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)

    # Test exception can be raised and caught
    try:
        raise UnknownExtension("Custom error message")
    except UnknownExtension as e:
        assert str(e) == "Custom error message"
    except CookiecutterException:
        pass  # Should also be catchable by parent

    # Test with empty string message
    exception = UnknownExtension("")
    assert str(exception) == ""


# LLM-generated content at query #45
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic exception creation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    message = "Invalid configuration file format"
    exception_with_msg = InvalidConfiguration(message)
    assert str(exception_with_msg) == message
    
    # Test exception inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    except CookiecutterException:
        pass  # Should also be catchable as parent type
    except Exception:
        pass  # Should also be catchable as base Exception type


# LLM-generated content at query #46
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""
    
    message = "Config file not found"
    exception_with_message = ConfigDoesNotExistException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #47
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Failed to decode JSON context"
    exception_with_msg = ContextDecodingException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert issubclass(ContextDecodingException, CookiecutterException)
    assert issubclass(ContextDecodingException, Exception)


# LLM-generated content at query #48
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    error = UndefinedError("'foo' is undefined")
    context = {"project_name": "Test Project", "author": "Test Author"}
    message = "Variable 'foo' was not defined in template"
    
    exception = UndefinedVariableInTemplate(message, error, context)
    
    expected_str = (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )
    
    assert str(exception) == expected_str
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context


# LLM-generated content at query #49
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic instantiation with no arguments
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    
    # Test instantiation with a longer message
    long_message = "The generated project directory was not found during cleanup"
    exception = MissingProjectDir(long_message)
    assert str(exception) == long_message
    
    # Test that it can be raised and caught
    try:
        raise MissingProjectDir("Test error")
    except MissingProjectDir as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    exception = MissingProjectDir()
    assert issubclass(MissingProjectDir, CookiecutterException)
    assert issubclass(MissingProjectDir, Exception)


# LLM-generated content at query #50
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception creation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(message)
    assert str(exception_with_message) == message
    
    # Test exception inheritance chain
    assert RepositoryNotFound.__mro__[0] is RepositoryNotFound
    assert RepositoryNotFound.__mro__[1] is CookiecutterException
    assert RepositoryNotFound.__mro__[2] is Exception
    
    # Test exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test exception can be caught by parent class
    try:
        raise RepositoryNotFound("Parent catch test")
    except CookiecutterException as e:
        assert str(e) == "Parent catch test"
    
    # Test exception can be caught by base Exception
    try:
        raise RepositoryNotFound("Base catch test")
    except Exception as e:
        assert str(e) == "Base catch test"


# LLM-generated content at query #51
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with default message
    exc = UnknownExtension()
    assert isinstance(exc, UnknownExtension)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Test extension not found"
    exc = UnknownExtension(custom_msg)
    assert str(exc) == custom_msg
    
    # Test exception chaining
    try:
        raise UnknownExtension("Original error")
    except UnknownExtension as e:
        assert str(e) == "Original error"
    
    # Test inheritance hierarchy
    assert UnknownExtension.__doc__ is not None
    assert "Exception for un-importable extension" in UnknownExtension.__doc__


# LLM-generated content at query #52
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Repository not found at specified location"
    exception = RepositoryNotFound(custom_message)
    assert str(exception) == custom_message
    
    # Test that exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert RepositoryNotFound.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #53
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic instantiation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Configuration file is malformed"
    exception_with_msg = InvalidConfiguration(message)
    assert str(exception_with_msg) == message
    
    # Test instantiation with complex message
    complex_msg = "Invalid YAML at line 5: unexpected indentation"
    exception_complex = InvalidConfiguration(complex_msg)
    assert str(exception_complex) == complex_msg
    
    # Test exception inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)


# LLM-generated content at query #54
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #55
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #56
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation with a message
    exception = ContextDecodingException("JSON decoding failed")
    assert str(exception) == "JSON decoding failed"
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with empty message
    exception = ContextDecodingException("")
    assert str(exception) == ""
    
    # Test instantiation with special characters in message
    message = "Failed to decode: {'key': 'value'}"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    
    # Test that it can be raised and caught
    try:
        raise ContextDecodingException("Test error")
    except ContextDecodingException as e:
        assert str(e) == "Test error"
    
    # Test exception hierarchy
    exception = ContextDecodingException("Test")
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #57
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic instantiation with no arguments
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Project directory not found at expected location"
    exception_with_msg = MissingProjectDir(message)
    assert str(exception_with_msg) == message
    
    # Test that it can be raised and caught
    try:
        raise MissingProjectDir("Test error")
    except MissingProjectDir as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert MissingProjectDir.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #58
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Test error message"
    exception_with_message = NonTemplatedInputDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #59
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #60
#--------------------------

```python
def test_CookiecutterException():
    # Test basic exception instantiation
    exception = CookiecutterException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, Exception)
    
    # Test exception with no message
    exception = CookiecutterException()
    assert str(exception) == ""
    
    # Test exception inheritance
    assert issubclass(CookiecutterException, Exception)
    
    # Test exception with custom attributes
    exception = CookiecutterException("Custom error")
    exception.custom_attr = "value"
    assert exception.custom_attr == "value"
    
    # Test exception chaining
    try:
        raise ValueError("Original error")
    except ValueError as e:
        exception = CookiecutterException("Wrapped error") from e
        assert exception.__cause__ is e


# LLM-generated content at query #61
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2 import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test case 1: Basic functionality
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "author": "Test Author"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    )
    assert str(exception) == expected_str
    
    # Test case 2: Empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "No variables defined",
        error,
        context
    )
    
    expected_str = "No variables defined. Error message: Missing variable. Context: {}"
    assert str(exception) == expected_str
    
    # Test case 3: Complex context with nested structures
    error = MockTemplateError("'user.profile.name' not found")
    context = {
        "user": {"id": 123, "email": "test@example.com"},
        "settings": {"theme": "dark", "notifications": True}
    }
    exception = UndefinedVariableInTemplate(
        "Nested variable access failed",
        error,
        context
    )
    
    expected_str = (
        "Nested variable access failed. "
        "Error message: 'user.profile.name' not found. "
        "Context: {'user': {'id': 123, 'email': 'test@example.com'}, "
        "'settings': {'theme': 'dark', 'notifications': True}}"
    )
    assert str(exception) == expected_str
    
    # Test case 4: Special characters in error message
    error = MockTemplateError("Variable 'test-data' (with-hyphens) is undefined")
    context = {"simple": "value"}
    exception = UndefinedVariableInTemplate(
        "Special character error",
        error,
        context
    )
    
    expected_str = (
        "Special character error. "
        "Error message: Variable 'test-data' (with-hyphens) is undefined. "
        "Context: {'simple': 'value'}"
    )
    assert str(exception) == expected_str
    
    # Test case 5: Verify attributes are stored correctly
    error = MockTemplateError("Test error")
    context = {"key": "value"}
    exception = UndefinedVariableInTemplate("Test message", error, context)
    
    assert exception.message == "Test message"
    assert exception.error == error
    assert exception.error.message == "Test error"
    assert exception.context == {"key": "value"}


# LLM-generated content at query #62
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic instantiation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Invalid configuration file format"
    exception_with_msg = InvalidConfiguration(message)
    assert str(exception_with_msg) == message
    
    # Test instantiation with a complex message
    complex_msg = "Invalid YAML at line 5: unexpected indentation"
    exception_complex = InvalidConfiguration(complex_msg)
    assert str(exception_complex) == complex_msg
    
    # Test that it can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert InvalidConfiguration.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #63
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #64
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic instantiation with no arguments
    exception = MissingProjectDir()
    assert isinstance(exception, MissingProjectDir)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Project directory not found at expected location"
    exception_with_message = MissingProjectDir(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise MissingProjectDir("Test error")
    except MissingProjectDir as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert MissingProjectDir.__doc__ is not None
    assert "Exception for missing generated project directory" in MissingProjectDir.__doc__


# LLM-generated content at query #65
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with message
    exc = UnknownExtension("Test extension error")
    assert str(exc) == "Test extension error"
    assert isinstance(exc, CookiecutterException)
    
    # Test instantiation without message
    exc2 = UnknownExtension()
    assert str(exc2) == ""
    
    # Test inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)
    
    # Test with custom attributes
    exc3 = UnknownExtension("Custom message")
    exc3.custom_attr = "value"
    assert exc3.custom_attr == "value"


# LLM-generated content at query #66
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    from jinja2 import TemplateError
    
    # Test basic initialization
    message = "Test message"
    error = TemplateError("Template error occurred")
    context = {"key": "value"}
    
    exception = UndefinedVariableInTemplate(message, error, context)
    
    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    
    # Test __str__ method
    str_representation = str(exception)
    assert message in str_representation
    assert "Template error occurred" in str_representation
    assert str(context) in str_representation
    
    # Test with different context types
    complex_context = {"nested": {"inner": "data"}, "list": [1, 2, 3]}
    exception2 = UndefinedVariableInTemplate("Another message", error, complex_context)
    assert exception2.context == complex_context
    
    # Test inheritance from CookiecutterException
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #67
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation with default message
    exc = InvalidModeException()
    assert isinstance(exc, InvalidModeException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Test error message"
    exc_with_msg = InvalidModeException(custom_msg)
    assert str(exc_with_msg) == custom_msg
    
    # Test exception inheritance chain
    assert issubclass(InvalidModeException, CookiecutterException)
    assert issubclass(InvalidModeException, Exception)
    
    # Test that exception can be raised and caught
    try:
        raise InvalidModeException("Test raising")
    except InvalidModeException as e:
        assert str(e) == "Test raising"
    
    # Test catching via parent class
    try:
        raise InvalidModeException("Parent catch test")
    except CookiecutterException as e:
        assert str(e) == "Parent catch test"


# LLM-generated content at query #68
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with default message
    exc1 = UnknownExtension()
    assert isinstance(exc1, UnknownExtension)
    assert isinstance(exc1, CookiecutterException)
    assert isinstance(exc1, Exception)
    assert str(exc1) == ""
    
    # Test instantiation with custom message
    custom_msg = "Custom extension error message"
    exc2 = UnknownExtension(custom_msg)
    assert str(exc2) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise UnknownExtension("Test error")
    except UnknownExtension as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert UnknownExtension.__mro__[0] is UnknownExtension
    assert UnknownExtension.__mro__[1] is CookiecutterException
    assert Exception in UnknownExtension.__mro__


# LLM-generated content at query #69
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic instantiation with no arguments
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Custom repository type error"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert UnknownRepoType.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #70
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Repository not found at specified location"
    exc = RepositoryNotFound(custom_msg)
    assert str(exc) == custom_msg
    
    # Test exception chaining
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance hierarchy
    assert RepositoryNotFound.__mro__[0] is RepositoryNotFound
    assert RepositoryNotFound.__mro__[1] is CookiecutterException
    assert RepositoryNotFound.__mro__[2] is Exception
    assert RepositoryNotFound.__mro__[3] is BaseException
    assert RepositoryNotFound.__mro__[4] is object


# LLM-generated content at query #71
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation with no arguments
    exc = InvalidZipRepository()
    assert isinstance(exc, InvalidZipRepository)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Invalid zip repository: corrupted archive"
    exc = InvalidZipRepository(message)
    assert exc.args == (message,)
    assert str(exc) == message

    # Test instantiation with multiple arguments
    exc = InvalidZipRepository("Error", "Additional info")
    assert exc.args == ("Error", "Additional info")
    assert str(exc) == "('Error', 'Additional info')"

    # Test inheritance chain
    assert InvalidZipRepository.__doc__ is not None
    assert "bad zip repo" in InvalidZipRepository.__doc__.lower()
    assert issubclass(InvalidZipRepository, CookiecutterException)
    assert issubclass(InvalidZipRepository, Exception)


# LLM-generated content at query #72
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""
    
    message = "Git is not installed"
    exception_with_message = VCSNotInstalled(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #73
#--------------------------

import pytest

def test_RepositoryCloneFailed():
    # Test instantiation without arguments
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Failed to clone repository"
    exc = RepositoryCloneFailed(message)
    assert exc.args == (message,)
    assert str(exc) == message
    
    # Test instantiation with multiple arguments
    exc = RepositoryCloneFailed("Error", "Additional info")
    assert exc.args == ("Error", "Additional info")
    assert str(exc) == "('Error', 'Additional info')"


# LLM-generated content at query #74
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic instantiation with no arguments
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have caught UnknownRepoType"
    
    # Test inheritance chain
    assert UnknownRepoType.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #75
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert exception.__class__.__name__ == "RepositoryNotFound"
    assert exception.__class__.__bases__[0].__name__ == "CookiecutterException"
    
    # Test exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by RepositoryNotFound"
    
    # Test multiple arguments
    exception_multi = RepositoryNotFound("Error", "additional", "args")
    assert str(exception_multi) == "('Error', 'additional', 'args')"


# LLM-generated content at query #76
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""

    # Test instantiation with message
    message = "Failed to decode JSON context"
    exception_with_msg = ContextDecodingException(message)
    assert str(exception_with_msg) == message

    # Test inheritance chain
    assert issubclass(ContextDecodingException, CookiecutterException)
    assert issubclass(ContextDecodingException, Exception)


# LLM-generated content at query #77
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Failed to decode JSON context"
    exception_with_msg = ContextDecodingException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert issubclass(ContextDecodingException, CookiecutterException)
    assert issubclass(ContextDecodingException, Exception)


# LLM-generated content at query #78
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation with no arguments
    exc1 = InvalidModeException()
    assert isinstance(exc1, InvalidModeException)
    assert isinstance(exc1, CookiecutterException)
    assert isinstance(exc1, Exception)
    assert str(exc1) == ""

    # Test instantiation with a message
    message = "Cannot use both no_input and replay modes simultaneously"
    exc2 = InvalidModeException(message)
    assert exc2.args == (message,)
    assert str(exc2) == message

    # Test instantiation with multiple arguments
    exc3 = InvalidModeException("Error", "Additional info")
    assert exc3.args == ("Error", "Additional info")
    assert str(exc3) == "('Error', 'Additional info')"

    # Test inheritance chain
    assert InvalidModeException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation with default message
    exc = EmptyDirNameException()
    assert isinstance(exc, EmptyDirNameException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Directory name cannot be empty"
    exc = EmptyDirNameException(custom_msg)
    assert str(exc) == custom_msg
    
    # Test exception inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    assert issubclass(EmptyDirNameException, CookiecutterException)


# LLM-generated content at query #2
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Failed to decode JSON context"
    exception_with_message = ContextDecodingException(message)
    assert exception_with_message.args[0] == message
    assert str(exception_with_message) == message


# LLM-generated content at query #3
#--------------------------

```python
def test_UnknownRepoType():
    # Test that UnknownRepoType is a subclass of CookiecutterException
    assert issubclass(UnknownRepoType, CookiecutterException)
    
    # Test instantiation with default no arguments
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test instantiation with multiple arguments
    multi_arg_exception = UnknownRepoType("arg1", "arg2", "arg3")
    assert str(multi_arg_exception) == "('arg1', 'arg2', 'arg3')"
    
    # Test that it can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    
    # Test exception chaining
    try:
        raise ValueError("Original error")
    except ValueError as e:
        repo_exception = UnknownRepoType("Repo error").with_traceback(e.__traceback__)
        assert str(repo_exception) == "Repo error"


# LLM-generated content at query #4
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with message
    exception = UnknownExtension("Test extension not found")
    assert str(exception) == "Test extension not found"
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with empty message
    exception = UnknownExtension("")
    assert str(exception) == ""
    
    # Test instantiation with multi-line message
    message = "Extension 'custom' could not be imported"
    exception = UnknownExtension(message)
    assert str(exception) == message
    
    # Test that it can be raised and caught
    try:
        raise UnknownExtension("Test error")
    except UnknownExtension as e:
        assert str(e) == "Test error"
    
    # Test exception chaining
    try:
        raise ImportError("Original error")
    except ImportError as e:
        try:
            raise UnknownExtension("Extension failed") from e
        except UnknownExtension as e2:
            assert str(e2) == "Extension failed"
            assert e2.__cause__ is e


# LLM-generated content at query #5
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test with simple values
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project"}
    exception = UndefinedVariableInTemplate(
        "Template variable error", error, context
    )
    
    expected_str = (
        "Template variable error. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "Another error", error, context
    )
    
    expected_str = (
        "Another error. "
        "Error message: Missing variable. "
        "Context: {}"
    )
    assert str(exception) == expected_str
    
    # Test with complex context
    error = MockTemplateError("'bar' not found")
    context = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    exception = UndefinedVariableInTemplate(
        "Complex error", error, context
    )
    
    expected_str = (
        "Complex error. "
        "Error message: 'bar' not found. "
        "Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3]}"
    )
    assert str(exception) == expected_str
    
    # Test that the exception attributes are properly set
    assert exception.message == "Complex error"
    assert exception.error.message == "'bar' not found"
    assert exception.context == {"nested": {"key": "value"}, "list": [1, 2, 3]}


# LLM-generated content at query #6
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test with simple values
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "version": "1.0"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'version': '1.0'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "Another error",
        error,
        context
    )
    
    expected_str = "Another error. Error message: Missing variable. Context: {}"
    assert str(exception) == expected_str
    
    # Test with complex context
    error = MockTemplateError("'bar' not found")
    context = {"nested": {"key": "value"}, "list": [1, 2, 3], "number": 42}
    exception = UndefinedVariableInTemplate(
        "Complex template error",
        error,
        context
    )
    
    expected_str = (
        "Complex template error. "
        "Error message: 'bar' not found. "
        "Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3], 'number': 42}"
    )
    assert str(exception) == expected_str
    
    # Test that the __str__ method properly formats all components
    error = MockTemplateError("Test error message")
    context = {"simple": "context"}
    exception = UndefinedVariableInTemplate(
        "Test main message",
        error,
        context
    )
    
    result = str(exception)
    assert "Test main message" in result
    assert "Test error message" in result
    assert "{'simple': 'context'}" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""

    message = "Test message"
    exception_with_message = NonTemplatedInputDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #8
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception creation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert exception.__class__.__name__ == "RepositoryNotFound"
    assert exception.__class__.__bases__[0].__name__ == "CookiecutterException"
    
    # Test that exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by RepositoryNotFound"
    
    # Test multiple inheritance levels
    try:
        raise RepositoryNotFound()
    except CookiecutterException:
        pass  # Should be caught by parent class
    except Exception:
        assert False, "Should have been caught by CookiecutterException"


# LLM-generated content at query #9
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation with message
    message = "Hook execution failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with empty message
    exception = FailedHookException("")
    assert str(exception) == ""
    
    # Test instantiation with complex message
    complex_message = "Hook 'pre_gen_project' failed with exit code 1"
    exception = FailedHookException(complex_message)
    assert str(exception) == complex_message
    
    # Test exception inheritance chain
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, FailedHookException)


# LLM-generated content at query #10
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation with no arguments
    exc = FailedHookException()
    assert isinstance(exc, FailedHookException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""

    # Test instantiation with a message
    message = "Hook script failed to execute"
    exc = FailedHookException(message)
    assert exc.args == (message,)
    assert str(exc) == message

    # Test instantiation with multiple arguments
    exc = FailedHookException("Hook failed", "pre_gen", "script.sh")
    assert len(exc.args) == 3
    assert exc.args[0] == "Hook failed"
    assert exc.args[1] == "pre_gen"
    assert exc.args[2] == "script.sh"

    # Test that it can be raised and caught
    try:
        raise FailedHookException("Test error")
    except FailedHookException as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by FailedHookException"

    # Test inheritance chain
    exc = FailedHookException()
    assert issubclass(FailedHookException, CookiecutterException)
    assert issubclass(FailedHookException, Exception)


# LLM-generated content at query #11
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    error = UndefinedError("'my_var' is undefined")
    context = {"project_name": "Test Project", "author": "Test Author"}
    exception = UndefinedVariableInTemplate(
        "Variable not found in template",
        error,
        context
    )
    
    expected_str = (
        "Variable not found in template. "
        "Error message: 'my_var' is undefined. "
        "Context: {'project_name': 'Test Project', 'author': 'Test Author'}"
    )
    
    assert str(exception) == expected_str
    
    error2 = UndefinedError("'another_var' is not defined")
    context2 = {}
    exception2 = UndefinedVariableInTemplate(
        "Missing variable",
        error2,
        context2
    )
    
    expected_str2 = (
        "Missing variable. "
        "Error message: 'another_var' is not defined. "
        "Context: {}"
    )
    
    assert str(exception2) == expected_str2
    
    error3 = UndefinedError("")
    context3 = {"key": "value"}
    exception3 = UndefinedVariableInTemplate(
        "Template error",
        error3,
        context3
    )
    
    expected_str3 = (
        "Template error. "
        "Error message: . "
        "Context: {'key': 'value'}"
    )
    
    assert str(exception3) == expected_str3


# LLM-generated content at query #12
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with message
    exc = UnknownExtension("Test extension error")
    assert str(exc) == "Test extension error"
    assert isinstance(exc, CookiecutterException)
    
    # Test instantiation without message
    exc2 = UnknownExtension()
    assert str(exc2) == ""
    
    # Test inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)
    
    # Test with custom attributes
    exc3 = UnknownExtension("Custom message")
    exc3.custom_attr = "value"
    assert exc3.custom_attr == "value"


# LLM-generated content at query #13
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test case 1: Basic functionality
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "version": "1.0"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'version': '1.0'}"
    )
    assert str(exception) == expected_str
    
    # Test case 2: Empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "No variables defined",
        error,
        context
    )
    
    expected_str = (
        "No variables defined. "
        "Error message: Missing variable. "
        "Context: {}"
    )
    assert str(exception) == expected_str
    
    # Test case 3: Complex context
    error = MockTemplateError("'user.email' not found")
    context = {
        "user": {"name": "John", "id": 123},
        "settings": {"debug": True, "log_level": "INFO"}
    }
    exception = UndefinedVariableInTemplate(
        "Configuration error",
        error,
        context
    )
    
    expected_str = (
        "Configuration error. "
        "Error message: 'user.email' not found. "
        "Context: {'user': {'name': 'John', 'id': 123}, 'settings': {'debug': True, 'log_level': 'INFO'}}"
    )
    assert str(exception) == expected_str
    
    # Test case 4: With actual Jinja2 UndefinedError
    try:
        # This would normally come from Jinja2 template rendering
        raise UndefinedError("'missing_var' is undefined")
    except UndefinedError as e:
        error = e
        context = {"available_var": "some_value"}
        exception = UndefinedVariableInTemplate(
            "Template rendering failed",
            error,
            context
        )
        
        # The actual error message might vary, so we check the structure
        result = str(exception)
        assert "Template rendering failed" in result
        assert "'missing_var' is undefined" in result
        assert "{'available_var': 'some_value'}" in result


# LLM-generated content at query #14
#--------------------------

def test_UndefinedVariableInTemplate():
    from jinja2 import UndefinedError
    error = UndefinedError("'foo' is undefined")
    context = {"bar": "baz"}
    exception = UndefinedVariableInTemplate(
        "Variable 'foo' is not defined", error, context
    )
    assert exception.message == "Variable 'foo' is not defined"
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        "Variable 'foo' is not defined. "
        "Error message: 'foo' is undefined. "
        "Context: {'bar': 'baz'}"
    )


# LLM-generated content at query #15
#--------------------------

```python
def test_UnknownTemplateDirException():
    # Test basic instantiation with no arguments
    exc1 = UnknownTemplateDirException()
    assert isinstance(exc1, UnknownTemplateDirException)
    assert isinstance(exc1, CookiecutterException)
    assert isinstance(exc1, Exception)
    assert str(exc1) == ""
    
    # Test instantiation with a message
    message = "Ambiguous template directory found"
    exc2 = UnknownTemplateDirException(message)
    assert str(exc2) == message
    assert exc2.args == (message,)
    
    # Test instantiation with multiple arguments
    exc3 = UnknownTemplateDirException("Error", "more info", 123)
    assert exc3.args == ("Error", "more info", 123)
    
    # Test that it can be raised and caught
    try:
        raise UnknownTemplateDirException("Test error")
    except UnknownTemplateDirException as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by UnknownTemplateDirException"
    
    # Test inheritance chain
    assert UnknownTemplateDirException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #16
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation with default message
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Directory name cannot be empty"
    exception = EmptyDirNameException(custom_message)
    assert str(exception) == custom_message
    
    # Test exception inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    
    # Test that exception can be raised and caught
    try:
        raise EmptyDirNameException("Test error")
    except EmptyDirNameException as e:
        assert str(e) == "Test error"
    
    # Test that it catches as parent class
    try:
        raise EmptyDirNameException()
    except CookiecutterException:
        assert True
    except Exception:
        assert True


# LLM-generated content at query #17
#--------------------------

import pytest


def test_FailedHookException():
    # Test basic instantiation with a message
    message = "Hook script failed to execute"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with no arguments
    exception = FailedHookException()
    assert str(exception) == ""
    
    # Test instantiation with custom arguments
    custom_message = "Custom hook failure"
    exception = FailedHookException(custom_message)
    assert str(exception) == custom_message


# LLM-generated content at query #18
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with a custom message
    custom_message = "Directory name cannot be empty"
    exception_with_message = EmptyDirNameException(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise EmptyDirNameException("Test error")
    except EmptyDirNameException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert EmptyDirNameException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)
    
    # Test empty instantiation has empty string representation
    assert str(exception) == ""


# LLM-generated content at query #19
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #20
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation with default message
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Test error message"
    exception = InvalidModeException(custom_message)
    assert str(exception) == custom_message
    
    # Test exception inheritance chain
    assert issubclass(InvalidModeException, CookiecutterException)
    assert issubclass(InvalidModeException, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidModeException("Test raise")
    except InvalidModeException as e:
        assert str(e) == "Test raise"
    except Exception:
        assert False, "Should have been caught by InvalidModeException"


# LLM-generated content at query #21
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic exception instantiation
    exception = InvalidConfiguration("Test error message")
    assert str(exception) == "Test error message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test exception with empty message
    exception = InvalidConfiguration("")
    assert str(exception) == ""
    
    # Test exception with complex message
    complex_msg = "Invalid config at line 5: missing required field 'name'"
    exception = InvalidConfiguration(complex_msg)
    assert str(exception) == complex_msg
    
    # Test exception inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidConfiguration("Configuration error")
    except InvalidConfiguration as e:
        assert str(e) == "Configuration error"
    except Exception:
        assert False, "Should have been caught by InvalidConfiguration"
    
    # Test multiple arguments (inherited from Exception base behavior)
    exception = InvalidConfiguration("Error", "additional", "args")
    assert exception.args == ("Error", "additional", "args")


# LLM-generated content at query #22
#--------------------------

```python
def test_UnknownTemplateDirException():
    exception = UnknownTemplateDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Test error message"
    exception_with_message = UnknownTemplateDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #23
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic instantiation
    exception = InvalidConfiguration("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test with empty message
    exception = InvalidConfiguration("")
    assert str(exception) == ""
    
    # Test with complex message
    complex_msg = "Invalid config: missing required field 'project_name'"
    exception = InvalidConfiguration(complex_msg)
    assert str(exception) == complex_msg
    
    # Test inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)


# LLM-generated content at query #24
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic instantiation with no arguments
    exc = MissingProjectDir()
    assert isinstance(exc, MissingProjectDir)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a message
    message = "Project directory not found at expected location"
    exc_with_msg = MissingProjectDir(message)
    assert exc_with_msg.args == (message,)
    assert str(exc_with_msg) == message
    
    # Test instantiation with multiple arguments
    multi_exc = MissingProjectDir("Error", "Additional info", 404)
    assert multi_exc.args == ("Error", "Additional info", 404)
    
    # Test exception inheritance chain
    assert MissingProjectDir.__doc__ is not None
    assert "missing generated project directory" in MissingProjectDir.__doc__
    assert issubclass(MissingProjectDir, CookiecutterException)
    assert issubclass(MissingProjectDir, Exception)


# LLM-generated content at query #25
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Repository not found at specified location"
    exception = RepositoryNotFound(custom_message)
    assert str(exception) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert RepositoryNotFound.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #26
#--------------------------

```python
def test_EmptyDirNameException():
    exception = EmptyDirNameException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Directory name cannot be empty"
    exception_with_message = EmptyDirNameException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #27
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with custom message
    custom_message = "Repository not found at specified location"
    exception = RepositoryNotFound(custom_message)
    assert str(exception) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert RepositoryNotFound.__base__ == CookiecutterException
    assert CookiecutterException.__base__ == Exception


# LLM-generated content at query #28
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation with default message
    exc = EmptyDirNameException()
    assert isinstance(exc, EmptyDirNameException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Directory name cannot be empty"
    exc = EmptyDirNameException(custom_msg)
    assert str(exc) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise EmptyDirNameException("Test error")
    except EmptyDirNameException as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    assert issubclass(EmptyDirNameException, CookiecutterException)


# LLM-generated content at query #29
#--------------------------

```python
def test_ContextDecodingException():
    exception = ContextDecodingException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Failed to decode JSON context"
    exception_with_message = ContextDecodingException(message)
    assert exception_with_message.args == (message,)
    assert str(exception_with_message) == message


# LLM-generated content at query #30
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, OutputDirExistsException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Output directory already exists"
    exception_with_msg = OutputDirExistsException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert OutputDirExistsException.__doc__ is not None
    assert "Exception for existing output directory" in OutputDirExistsException.__doc__
    
    # Test that it can be raised and caught
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
    
    # Test that it catches as parent exception
    try:
        raise OutputDirExistsException()
    except CookiecutterException:
        assert True
    except Exception:
        assert True


# LLM-generated content at query #31
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Git is not installed"
    exception_with_message = VCSNotInstalled(message)
    assert str(exception_with_message) == message
    assert exception_with_message.args == (message,)


# LLM-generated content at query #32
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository not found at specified location"
    exception_with_message = RepositoryNotFound(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert issubclass(RepositoryNotFound, CookiecutterException)
    assert issubclass(RepositoryNotFound, Exception)
    
    # Test exception can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test multiple inheritance levels
    exception = RepositoryNotFound("Another test")
    assert exception.__class__.__name__ == "RepositoryNotFound"
    assert exception.__class__.__bases__[0].__name__ == "CookiecutterException"


# LLM-generated content at query #33
#--------------------------

```python
def test_ContextDecodingException():
    # Test basic instantiation
    exception = ContextDecodingException()
    assert isinstance(exception, ContextDecodingException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with message
    message = "Test error message"
    exception_with_msg = ContextDecodingException(message)
    assert str(exception_with_msg) == message
    
    # Test that it can be raised and caught
    try:
        raise ContextDecodingException("JSON decoding failed")
    except ContextDecodingException as e:
        assert str(e) == "JSON decoding failed"
    except Exception:
        assert False, "Should have been caught by ContextDecodingException"
    
    # Test inheritance chain
    assert ContextDecodingException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #34
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation
    exception = InvalidModeException()
    assert isinstance(exception, InvalidModeException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Test error message"
    exception_with_msg = InvalidModeException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert issubclass(InvalidModeException, CookiecutterException)
    assert issubclass(InvalidModeException, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidModeException("Test raising")
    except InvalidModeException as e:
        assert str(e) == "Test raising"
    
    # Test multiple inheritance levels
    try:
        raise InvalidModeException("Another test")
    except CookiecutterException:
        pass  # Should be caught by parent class
    
    # Test empty message
    empty_exception = InvalidModeException("")
    assert str(empty_exception) == ""


# LLM-generated content at query #35
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic exception instantiation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with a message
    message = "Configuration file is malformed"
    exception_with_msg = InvalidConfiguration(message)
    assert str(exception_with_msg) == message
    
    # Test exception inheritance chain
    assert exception.__class__.__name__ == "InvalidConfiguration"
    assert exception.__class__.__bases__[0].__name__ == "CookiecutterException"
    
    # Test exception can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by InvalidConfiguration"


# LLM-generated content at query #36
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic exception instantiation
    exception = RepositoryNotFound()
    assert isinstance(exception, RepositoryNotFound)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with message
    message = "Repository not found at specified location"
    exception_with_msg = RepositoryNotFound(message)
    assert exception_with_msg.args == (message,)
    assert str(exception_with_msg) == message
    
    # Test exception with multiple arguments
    multi_arg_exception = RepositoryNotFound("Error", "additional", "info")
    assert multi_arg_exception.args == ("Error", "additional", "info")
    assert str(multi_arg_exception) == "('Error', 'additional', 'info')"
    
    # Test exception inheritance chain
    assert issubclass(RepositoryNotFound, CookiecutterException)
    assert issubclass(RepositoryNotFound, Exception)


# LLM-generated content at query #37
#--------------------------

```python
def test_InvalidModeException():
    # Test basic instantiation with default message
    exc = InvalidModeException()
    assert isinstance(exc, InvalidModeException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Test error message"
    exc = InvalidModeException(custom_msg)
    assert str(exc) == custom_msg
    
    # Test exception inheritance chain
    assert issubclass(InvalidModeException, CookiecutterException)
    assert issubclass(InvalidModeException, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidModeException("Test raise")
    except InvalidModeException as e:
        assert str(e) == "Test raise"
    except Exception:
        assert False, "Should have been caught by InvalidModeException"


# LLM-generated content at query #38
#--------------------------

def test_FailedHookException():
    exception = FailedHookException("Hook script failed")
    assert str(exception) == "Hook script failed"
    assert isinstance(exception, CookiecutterException)


# LLM-generated content at query #39
#--------------------------

def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    
    exception_with_cause = RepositoryCloneFailed("Clone failed") from ValueError("Invalid URL")
    assert str(exception_with_cause) == "Clone failed"
    assert exception_with_cause.__cause__ is not None
    assert isinstance(exception_with_cause.__cause__, ValueError)


# LLM-generated content at query #40
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic exception instantiation
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Repository type could not be determined"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test exception inheritance chain
    assert UnknownRepoType.__doc__ is not None
    assert "Exception for unknown repo types." in UnknownRepoType.__doc__
    assert UnknownRepoType.__bases__ == (CookiecutterException,)


# LLM-generated content at query #41
#--------------------------

def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Clone failed")
    assert str(exception) == "Clone failed"
    assert isinstance(exception, CookiecutterException)
    
    exception_with_custom_message = RepositoryCloneFailed("Custom error message")
    assert str(exception_with_custom_message) == "Custom error message"


# LLM-generated content at query #42
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic exception creation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Invalid configuration file format"
    exception_with_msg = InvalidConfiguration(custom_message)
    assert str(exception_with_msg) == custom_message
    
    # Test exception inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by InvalidConfiguration"
    
    # Test multiple arguments
    exception_multi = InvalidConfiguration("Error", "in", "config")
    assert str(exception_multi) == "('Error', 'in', 'config')"


# LLM-generated content at query #43
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    from jinja2.exceptions import UndefinedError
    from cookiecutter.exceptions import UndefinedVariableInTemplate

    error = UndefinedError("Variable 'foo' is undefined")
    context = {"bar": "baz", "num": 42}
    message = "Template variable error occurred"

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == (
        f"{message}. "
        f"Error message: {error.message}. "
        f"Context: {context}"
    )


# LLM-generated content at query #44
#--------------------------

```python
def test_ConfigDoesNotExistException():
    # Test basic exception instantiation
    exc = ConfigDoesNotExistException()
    assert isinstance(exc, ConfigDoesNotExistException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test exception with custom message
    message = "Config file not found at /path/to/config.yaml"
    exc_with_msg = ConfigDoesNotExistException(message)
    assert str(exc_with_msg) == message
    
    # Test exception inheritance chain
    assert ConfigDoesNotExistException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #45
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic instantiation with default message
    exc1 = RepositoryCloneFailed()
    assert isinstance(exc1, RepositoryCloneFailed)
    assert isinstance(exc1, CookiecutterException)
    assert str(exc1) == ""
    
    # Test instantiation with custom message
    custom_msg = "Failed to clone repository from https://example.com"
    exc2 = RepositoryCloneFailed(custom_msg)
    assert str(exc2) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise RepositoryCloneFailed("Clone error")
    except RepositoryCloneFailed as e:
        assert str(e) == "Clone error"
    
    # Test inheritance chain
    assert RepositoryCloneFailed.__mro__[0] is RepositoryCloneFailed
    assert RepositoryCloneFailed.__mro__[1] is CookiecutterException
    assert RepositoryCloneFailed.__mro__[2] is Exception


# LLM-generated content at query #46
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic exception instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_msg = "Output directory already exists at /path/to/dir"
    exception_with_msg = OutputDirExistsException(custom_msg)
    assert str(exception_with_msg) == custom_msg
    
    # Test exception inheritance chain
    assert issubclass(OutputDirExistsException, CookiecutterException)
    assert issubclass(OutputDirExistsException, Exception)
    
    # Test exception can be raised and caught
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
    except Exception:
        assert False, "Should have caught OutputDirExistsException"


# LLM-generated content at query #47
#--------------------------

```python
def test_UnknownExtension():
    # Test basic instantiation with a message
    exception = UnknownExtension("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)

    # Test instantiation without message
    exception = UnknownExtension()
    assert str(exception) == ""

    # Test that it can be raised and caught
    try:
        raise UnknownExtension("Custom error")
    except UnknownExtension as e:
        assert str(e) == "Custom error"

    # Test inheritance chain
    assert issubclass(UnknownExtension, CookiecutterException)
    assert issubclass(UnknownExtension, Exception)


# LLM-generated content at query #48
#--------------------------

```python
def test_RepositoryNotFound():
    # Test basic instantiation with default message
    exc = RepositoryNotFound()
    assert isinstance(exc, RepositoryNotFound)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Repository not found at specified location"
    exc = RepositoryNotFound(custom_msg)
    assert str(exc) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise RepositoryNotFound("Test error")
    except RepositoryNotFound as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert RepositoryNotFound.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #49
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with custom message
    custom_message = "Directory name cannot be empty"
    exception_with_message = EmptyDirNameException(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise EmptyDirNameException("Test exception")
    except EmptyDirNameException as e:
        assert str(e) == "Test exception"
    
    # Test inheritance chain
    assert EmptyDirNameException.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #50
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation with no arguments
    exc = OutputDirExistsException()
    assert isinstance(exc, OutputDirExistsException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a custom message
    message = "Output directory already exists at /path/to/dir"
    exc = OutputDirExistsException(message)
    assert exc.args == (message,)
    assert str(exc) == message
    
    # Test that it can be raised and caught
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
    
    # Test inheritance chain
    assert issubclass(OutputDirExistsException, CookiecutterException)
    assert issubclass(OutputDirExistsException, Exception)


# LLM-generated content at query #51
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation with default message
    exc = EmptyDirNameException()
    assert isinstance(exc, EmptyDirNameException)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with custom message
    custom_msg = "Directory name cannot be empty"
    exc = EmptyDirNameException(custom_msg)
    assert str(exc) == custom_msg
    
    # Test exception inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    
    # Test exception can be raised and caught
    try:
        raise EmptyDirNameException("Test error")
    except EmptyDirNameException as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have caught EmptyDirNameException"


# LLM-generated content at query #52
#--------------------------

```python
def test_RepositoryCloneFailed():
    # Test basic exception instantiation
    exc = RepositoryCloneFailed()
    assert isinstance(exc, RepositoryCloneFailed)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test exception with custom message
    message = "Failed to clone repository from https://example.com/template.git"
    exc_with_message = RepositoryCloneFailed(message)
    assert exc_with_message.args == (message,)
    assert str(exc_with_message) == message
    
    # Test exception with multiple arguments
    multi_arg_exc = RepositoryCloneFailed("Clone failed", "Network error", 404)
    assert multi_arg_exc.args == ("Clone failed", "Network error", 404)
    
    # Test exception inheritance chain
    assert issubclass(RepositoryCloneFailed, CookiecutterException)
    assert issubclass(RepositoryCloneFailed, Exception)
    
    # Test exception can be raised and caught
    try:
        raise RepositoryCloneFailed("Test error")
    except RepositoryCloneFailed as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by RepositoryCloneFailed"


# LLM-generated content at query #53
#--------------------------

```python
def test_RepositoryNotFound():
    exception = RepositoryNotFound("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #54
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic instantiation with no arguments
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Custom error message"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert UnknownRepoType.__bases__[0] is CookiecutterException
    assert CookiecutterException.__bases__[0] is Exception


# LLM-generated content at query #55
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test with simple values
    error = MockTemplateError("Variable 'foo' is undefined")
    context = {"project_name": "Test Project", "author": "John Doe"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'Test Project', 'author': 'John Doe'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "Another error",
        error,
        context
    )
    
    expected_str = "Another error. Error message: Missing variable. Context: {}"
    assert str(exception) == expected_str
    
    # Test with complex context
    error = MockTemplateError("'bar' not found")
    context = {"nested": {"key": "value"}, "list": [1, 2, 3], "number": 42}
    exception = UndefinedVariableInTemplate(
        "Complex template error",
        error,
        context
    )
    
    expected_str = (
        "Complex template error. "
        "Error message: 'bar' not found. "
        "Context: {'nested': {'key': 'value'}, 'list': [1, 2, 3], 'number': 42}"
    )
    assert str(exception) == expected_str
    
    # Test with special characters in message
    error = MockTemplateError("Variable 'user-input' contains & special <chars>")
    context = {"test": "data"}
    exception = UndefinedVariableInTemplate(
        "Special char error",
        error,
        context
    )
    
    expected_str = (
        "Special char error. "
        "Error message: Variable 'user-input' contains & special <chars>. "
        "Context: {'test': 'data'}"
    )
    assert str(exception) == expected_str


# LLM-generated content at query #56
#--------------------------

```python
def test_InvalidModeException():
    exception = InvalidModeException("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #57
#--------------------------

```python
def test_InvalidZipRepository():
    # Test that InvalidZipRepository can be instantiated without arguments
    exception1 = InvalidZipRepository()
    assert isinstance(exception1, InvalidZipRepository)
    assert isinstance(exception1, CookiecutterException)
    assert isinstance(exception1, Exception)
    assert str(exception1) == ""

    # Test that InvalidZipRepository can be instantiated with a message
    message = "Invalid zip repository format"
    exception2 = InvalidZipRepository(message)
    assert isinstance(exception2, InvalidZipRepository)
    assert str(exception2) == message

    # Test that InvalidZipRepository can be instantiated with a message and custom attributes
    exception3 = InvalidZipRepository("Bad zip file", "extra_info")
    assert str(exception3) == "Bad zip file"

    # Test inheritance chain
    assert InvalidZipRepository.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #58
#--------------------------

```python
def test_CookiecutterException():
    # Test basic instantiation with no arguments
    exception = CookiecutterException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Test error message"
    exception_with_msg = CookiecutterException(message)
    assert str(exception_with_msg) == message
    
    # Test instantiation with multiple arguments
    multi_arg_exception = CookiecutterException("Error", "code", 500)
    assert str(multi_arg_exception) == "('Error', 'code', 500)"
    
    # Test that it can be raised and caught
    try:
        raise CookiecutterException("Test raising")
    except CookiecutterException as e:
        assert str(e) == "Test raising"
    
    # Test inheritance chain
    assert CookiecutterException.__base__ == Exception
    assert CookiecutterException.__name__ == "CookiecutterException"


# LLM-generated content at query #59
#--------------------------

```python
def test_VCSNotInstalled():
    exception = VCSNotInstalled()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""

    message = "Git is not installed"
    exception_with_message = VCSNotInstalled(message)
    assert str(exception_with_message) == message
    assert exception_with_message.args == (message,)


# LLM-generated content at query #60
#--------------------------

```python
def test_EmptyDirNameException():
    # Test basic instantiation
    exception = EmptyDirNameException()
    assert isinstance(exception, EmptyDirNameException)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    
    # Test instantiation with custom message
    custom_message = "Directory name cannot be empty"
    exception_with_message = EmptyDirNameException(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise EmptyDirNameException("Test exception")
    except EmptyDirNameException as e:
        assert str(e) == "Test exception"
    
    # Test inheritance chain
    assert EmptyDirNameException.__doc__ is not None
    assert "Exception for a empty directory name" in EmptyDirNameException.__doc__
    
    # Test empty string message
    empty_msg_exception = EmptyDirNameException("")
    assert str(empty_msg_exception) == ""


# LLM-generated content at query #61
#--------------------------

```python
def test_OutputDirExistsException():
    # Test basic instantiation
    exception = OutputDirExistsException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Output directory already exists"
    exception_with_msg = OutputDirExistsException(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert issubclass(OutputDirExistsException, CookiecutterException)
    assert issubclass(OutputDirExistsException, Exception)
    
    # Test exception can be raised and caught
    try:
        raise OutputDirExistsException("Test message")
    except OutputDirExistsException as e:
        assert str(e) == "Test message"
    except Exception:
        assert False, "Should have been caught by OutputDirExistsException"


# LLM-generated content at query #62
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    custom_message = "Config file not found at /path/to/config"
    exception_with_message = ConfigDoesNotExistException(custom_message)
    assert str(exception_with_message) == custom_message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #63
#--------------------------

```python
def test_UnknownRepoType():
    # Test basic instantiation with no arguments
    exception = UnknownRepoType()
    assert isinstance(exception, UnknownRepoType)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a custom message
    custom_message = "Custom repository type error"
    exception_with_message = UnknownRepoType(custom_message)
    assert str(exception_with_message) == custom_message
    
    # Test that it can be raised and caught
    try:
        raise UnknownRepoType("Test error")
    except UnknownRepoType as e:
        assert str(e) == "Test error"
    
    # Test inheritance chain
    assert UnknownRepoType.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #64
#--------------------------

```python
def test_ConfigDoesNotExistException():
    exception = ConfigDoesNotExistException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Config file not found at /path/to/config.yaml"
    exception_with_message = ConfigDoesNotExistException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #65
#--------------------------

```python
def test_UndefinedVariableInTemplate():
    # Test basic initialization with all required parameters
    error = TemplateError("Variable 'foo' is undefined")
    context = {"project_name": "test_project", "author": "test_author"}
    exception = UndefinedVariableInTemplate(
        "Template variable error", error, context
    )
    
    assert exception.message == "Template variable error"
    assert exception.error == error
    assert exception.error.message == "Variable 'foo' is undefined"
    assert exception.context == context
    
    # Test string representation
    expected_str = (
        "Template variable error. "
        "Error message: Variable 'foo' is undefined. "
        "Context: {'project_name': 'test_project', 'author': 'test_author'}"
    )
    assert str(exception) == expected_str
    
    # Test with empty context
    exception2 = UndefinedVariableInTemplate(
        "Another error", TemplateError("Missing variable"), {}
    )
    assert str(exception2) == "Another error. Error message: Missing variable. Context: {}"
    
    # Test with complex error message
    complex_error = TemplateError("'bar' is not defined in this context")
    complex_context = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    exception3 = UndefinedVariableInTemplate(
        "Complex template error", complex_error, complex_context
    )
    
    assert exception3.error.message == "'bar' is not defined in this context"
    assert exception3.context == complex_context


# LLM-generated content at query #66
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation with message
    exception = FailedHookException("Hook execution failed")
    assert str(exception) == "Hook execution failed"
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with empty message
    exception = FailedHookException("")
    assert str(exception) == ""
    
    # Test instantiation with complex message
    message = "Pre-gen hook failed with exit code 1"
    exception = FailedHookException(message)
    assert str(exception) == message
    
    # Test inheritance chain
    assert issubclass(FailedHookException, CookiecutterException)
    assert issubclass(FailedHookException, Exception)


# LLM-generated content at query #67
#--------------------------

def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Test message")
    assert str(exception) == "Test message"
    assert isinstance(exception, CookiecutterException)
    assert exception.args == ("Test message",)


# LLM-generated content at query #68
#--------------------------

```python
def test_VCSNotInstalled():
    # Test basic instantiation with no arguments
    exc = VCSNotInstalled()
    assert isinstance(exc, VCSNotInstalled)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test instantiation with a custom message
    custom_msg = "Git is not installed on this system"
    exc_with_msg = VCSNotInstalled(custom_msg)
    assert str(exc_with_msg) == custom_msg
    
    # Test that it can be raised and caught
    try:
        raise VCSNotInstalled("Test message")
    except VCSNotInstalled as e:
        assert str(e) == "Test message"
    
    # Test inheritance chain
    assert VCSNotInstalled.__bases__ == (CookiecutterException,)
    assert CookiecutterException.__bases__ == (Exception,)


# LLM-generated content at query #69
#--------------------------

```python
def test_NonTemplatedInputDirException():
    exception = NonTemplatedInputDirException()
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    message = "Input directory should be templated"
    exception_with_message = NonTemplatedInputDirException(message)
    assert str(exception_with_message) == message
    assert isinstance(exception_with_message, CookiecutterException)


# LLM-generated content at query #70
#--------------------------

```python
def test_MissingProjectDir():
    # Test basic exception instantiation
    exc = MissingProjectDir()
    assert isinstance(exc, MissingProjectDir)
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
    assert str(exc) == ""
    
    # Test exception with custom message
    message = "Project directory not found at expected location"
    exc_with_msg = MissingProjectDir(message)
    assert str(exc_with_msg) == message
    
    # Test exception inheritance chain
    assert MissingProjectDir.__doc__ is not None
    assert "Exception for missing generated project directory" in MissingProjectDir.__doc__
    
    # Test exception can be raised and caught
    try:
        raise MissingProjectDir("Test error")
    except MissingProjectDir as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by MissingProjectDir"
    
    # Test exception with empty string message
    exc_empty = MissingProjectDir("")
    assert str(exc_empty) == ""
    
    # Test exception with None message (should convert to empty string)
    exc_none = MissingProjectDir(None)
    assert str(exc_none) == "None"


# LLM-generated content at query #71
#--------------------------

def test_RepositoryCloneFailed():
    exception = RepositoryCloneFailed("Clone failed due to network error")
    assert str(exception) == "Clone failed due to network error"
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)


# LLM-generated content at query #72
#--------------------------

```python
def test_InvalidConfiguration():
    # Test basic exception instantiation
    exception = InvalidConfiguration()
    assert isinstance(exception, InvalidConfiguration)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test exception with custom message
    custom_message = "Configuration file is malformed"
    exception_with_msg = InvalidConfiguration(custom_message)
    assert str(exception_with_msg) == custom_message
    
    # Test exception inheritance chain
    assert issubclass(InvalidConfiguration, CookiecutterException)
    assert issubclass(InvalidConfiguration, Exception)
    
    # Test exception can be raised and caught
    try:
        raise InvalidConfiguration("Test error")
    except InvalidConfiguration as e:
        assert str(e) == "Test error"
    
    # Test multiple inheritance levels
    try:
        raise InvalidConfiguration("Another test")
    except CookiecutterException:
        assert True
    except Exception:
        assert True


# LLM-generated content at query #73
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with message
    message = "Invalid zip file format"
    exception_with_msg = InvalidZipRepository(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert InvalidZipRepository.__doc__ is not None
    assert "Exception for bad zip repo" in InvalidZipRepository.__doc__
    
    # Test exception can be raised and caught
    try:
        raise InvalidZipRepository("Test error")
    except InvalidZipRepository as e:
        assert str(e) == "Test error"
    except Exception:
        assert False, "Should have been caught by InvalidZipRepository"


# LLM-generated content at query #74
#--------------------------

```python
def test_InvalidZipRepository():
    # Test basic instantiation
    exception = InvalidZipRepository()
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == ""
    
    # Test instantiation with a message
    message = "Invalid zip file format"
    exception_with_msg = InvalidZipRepository(message)
    assert str(exception_with_msg) == message
    
    # Test inheritance chain
    assert InvalidZipRepository.__doc__ is not None
    assert "Exception for bad zip repo" in InvalidZipRepository.__doc__
    
    # Test exception can be raised and caught
    try:
        raise InvalidZipRepository("Test message")
    except InvalidZipRepository as e:
        assert str(e) == "Test message"
    
    # Test it can be caught by parent exceptions
    try:
        raise InvalidZipRepository()
    except CookiecutterException:
        assert True
    except Exception:
        assert True


# LLM-generated content at query #75
#--------------------------

```python
def test_UndefinedVariableInTemplate___str__():
    from jinja2.exceptions import UndefinedError
    
    # Create a mock TemplateError with a message attribute
    class MockTemplateError:
        def __init__(self, message):
            self.message = message
    
    # Test case 1: Basic functionality
    error = MockTemplateError("Variable 'username' is undefined")
    context = {"project_name": "Test Project", "version": "1.0"}
    exception = UndefinedVariableInTemplate(
        "Template variable error occurred",
        error,
        context
    )
    
    expected_str = (
        "Template variable error occurred. "
        "Error message: Variable 'username' is undefined. "
        "Context: {'project_name': 'Test Project', 'version': '1.0'}"
    )
    assert str(exception) == expected_str
    
    # Test case 2: Empty context
    error = MockTemplateError("Missing variable")
    context = {}
    exception = UndefinedVariableInTemplate(
        "No variables defined",
        error,
        context
    )
    
    expected_str = (
        "No variables defined. "
        "Error message: Missing variable. "
        "Context: {}"
    )
    assert str(exception) == expected_str
    
    # Test case 3: Complex context with nested structures
    error = MockTemplateError("'user.profile.name' not found")
    context = {
        "settings": {"debug": True, "port": 8000},
        "users": ["alice", "bob"],
        "metadata": None
    }
    exception = UndefinedVariableInTemplate(
        "Complex template error",
        error,
        context
    )
    
    expected_str = (
        "Complex template error. "
        "Error message: 'user.profile.name' not found. "
        "Context: {'settings': {'debug': True, 'port': 8000}, 'users': ['alice', 'bob'], 'metadata': None}"
    )
    assert str(exception) == expected_str
    
    # Test case 4: Verify attributes are stored correctly
    error = MockTemplateError("Test error")
    context = {"test": "value"}
    exception = UndefinedVariableInTemplate(
        "Test message",
        error,
        context
    )
    
    assert exception.message == "Test message"
    assert exception.error == error
    assert exception.error.message == "Test error"
    assert exception.context == {"test": "value"}


# LLM-generated content at query #76
#--------------------------

```python
def test_FailedHookException():
    # Test basic instantiation with message
    message = "Hook execution failed"
    exception = FailedHookException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
    
    # Test instantiation with empty message
    empty_exception = FailedHookException("")
    assert str(empty_exception) == ""
    
    # Test instantiation with complex message
    complex_message = "Hook 'pre_gen_project' failed with exit code 1"
    complex_exception = FailedHookException(complex_message)
    assert str(complex_exception) == complex_message
    
    # Test exception inheritance chain
    assert issubclass(FailedHookException, CookiecutterException)
    assert issubclass(FailedHookException, Exception)


