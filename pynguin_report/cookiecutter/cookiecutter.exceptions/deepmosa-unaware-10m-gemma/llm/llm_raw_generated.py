####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mock the TemplateError and its message attribute
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format: "{message}. Error message: {error.message}. Context: {context}"
    expected_str = "Variable not found. Error message: undefined variable 'foo'. Context: {'project_name': 'my_project'}"
    
    # Act
    actual_str = str(exception)
    
    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    """
    Test that InvalidZipRepository can be instantiated and inherits 
    from CookiecutterException.
    """
    message = "The provided zip repository is invalid."
    
    # Test instantiation
    exception = InvalidZipRepository(message)
    
    # Verify message content
    assert str(exception) == message
    
    # Verify inheritance hierarchy
    assert isinstance(exception, InvalidZipRepository)
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError from jinja2
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format:
    # "{self.message}. Error message: {self.error.message}. Context: {self.context}"
    expected_output = (
        "Variable not found. "
        "Error message: undefined variable 'foo'. "
        "Context: {'project_name': 'my_project'}"
    )

    # Act
    actual_output = str(exception)

    # Assert
    assert actual_output == expected_output
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    """Test that InvalidZipRepository can be instantiated and inherits correctly."""
    message = "Invalid zip file"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert isinstance(exception, Exception)
    assert str(exception) == message
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #9
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exc = MissingProjectDir(message)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception test"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is used in the exception
    mock_error = MagicMock()
    mock_error.message = "Undefined symbol 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )

    # Act
    actual_str = str(exception)

    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError from jinja2
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #19
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base error message"
    exc = CookiecutterException(message)
    
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #25
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML configuration"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception test"
    exception = CookiecutterException(message)
    
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #32
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking jinja2.TemplateError as it is used in the constructor
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #35
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #36
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repo not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #38
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Test message"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #39
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #40
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #41
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #42
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Test error message"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #44
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #46
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repo not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #47
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is expected to have a .message attribute
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'user' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Act
    result = str(exception)
    
    # Assert
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert result == expected_str
```


# LLM-generated content at query #49
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup mock TemplateError since it's an external dependency
    mock_error = MagicMock()
    mock_error.message = "Undefined variable 'project_name'"
    
    message = "Template error occurred"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Initialize the exception
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format: "{message}. Error message: {error.message}. Context: {context}"
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    
    assert str(exception) == expected_str
```


# LLM-generated content at query #51
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #53
#--------------------------

```python
def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #54
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    context = {"project_name": "test_project", "version": "1.0.0"}

    # Execution
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )

    # Assertions
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test __str__ implementation
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #56
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exception = CookiecutterException(message)
    
    assert str(exception) == message
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #57
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #58
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #59
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #60
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "jinja2 error message"
    context = {"project_name": "my_project", "version": "1.0.0"}

    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )

    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #61
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #62
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #63
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #64
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #65
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #66
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception test"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #67
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #68
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #69
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #70
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #71
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #72
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #73
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #74
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #75
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #76
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #77
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #78
#--------------------------

```python
import pytest

def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #79
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #80
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Mock TemplateError from jinja2
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    message = "A template error occurred"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    exception = UndefinedVariableInTemplate(
        message=message,
        error=mock_error,
        context=context
    )
    
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    
    assert str(exception) == expected_str
```


# LLM-generated content at query #81
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    """
    Test that InvalidZipRepository can be instantiated and behaves as a subclass 
    of CookiecutterException.
    """
    message = "The provided zip file is not a valid archive."
    
    # Test instantiation
    exc = InvalidZipRepository(message)
    
    # Assertions
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #82
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #83
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #84
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #85
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #86
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #87
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #88
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #89
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #90
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #91
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #92
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #93
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #94
#--------------------------

```python
import pytest

def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #95
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #96
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #97
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #98
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #99
#--------------------------

```python
def test_UnknownExtension():
    message = "Unknown extension error"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #100
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #101
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #102
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #103
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #104
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Setup mock dependencies
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "jinja2 error message"
    context = {"project_name": "my_project", "version": "1.0.0"}

    # Instantiate the exception
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )

    # Assert attributes are correctly assigned
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context

    # Assert the string representation (str method) is formatted correctly
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #105
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #106
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #107
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #108
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #109
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #110
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #111
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #112
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #113
#--------------------------

```python
import pytest

def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #114
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #115
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #116
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception message"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #117
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #118
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #119
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension error"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #120
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Create a mock for TemplateError
    mock_error = MagicMock()
    mock_error.message = "Undefined symbol 'foo'"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test the __str__ implementation
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #121
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #122
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #123
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #124
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #125
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #126
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #127
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking Jinja2 TemplateError
    mock_error = MagicMock()
    mock_error.message = "Undefined symbol 'foo'"
    
    # Execution
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assertions
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test __str__ implementation
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #128
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #129
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #130
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is an external dependency
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format based on implementation:
    # "{self.message}. Error message: {self.error.message}. Context: {self.context}"
    expected_output = (
        "Variable not found. "
        "Error message: UndefinedError: 'foo' is undefined. "
        "Context: {'project_name': 'my_project'}"
    )
    
    # Act
    actual_output = str(exception)
    
    # Assert
    assert actual_output == expected_output
```


# LLM-generated content at query #131
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #132
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #133
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #134
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Failed to unzip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #135
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #136
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "A hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #137
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #138
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "The input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #139
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #140
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #141
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #142
#--------------------------

```python
import pytest

def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #143
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #144
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #145
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension error"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #146
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #147
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #148
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #149
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #150
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #151
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #152
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #153
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #154
#--------------------------

```python
def test_UnknownExtension():
    message = "Test message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #155
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #156
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #157
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #158
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup mock for TemplateError
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    message = "An error occurred"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Instantiate the exception
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format: "{message}. Error message: {error.message}. Context: {context}"
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    
    # Assertions
    assert str(exception) == expected_str
```


# LLM-generated content at query #159
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "The provided repository is not a valid zip file."
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #160
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Create a mock for TemplateError which has a 'message' attribute
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    expected_str = (
        "Variable not found. "
        "Error message: UndefinedError: 'foo' is undefined. "
        "Context: {'project_name': 'my_project'}"
    )
    
    # Act
    actual_str = str(exception)
    
    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #161
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #162
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #163
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #164
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #165
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #166
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #167
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #168
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #169
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #170
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #171
#--------------------------

```python
import pytest

def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #172
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mock the TemplateError and its message attribute
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format: "{message}. Error message: {error.message}. Context: {context}"
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    
    # Assert
    assert str(exception) == expected_str
```


# LLM-generated content at query #173
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #174
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #175
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #176
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #177
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #178
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes provided"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #179
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exc = CookiecutterException(message)
    
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #180
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #2
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #3
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #4
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError (which is imported via TYPE_CHECKING)
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format: "{message}. Error message: {error.message}. Context: {context}"
    expected_output = f"{message}. Error message: {mock_error.message}. Context: {context}"
    
    # Act
    actual_output = str(exception)
    
    # Assert
    assert actual_output == expected_output
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "The input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Directory already exists"
    with pytest.raises(OutputDirExistsException) as excinfo:
        raise OutputDirExistsException(message)
    
    assert str(excinfo.value) == message
    assert isinstance(excinfo.value, CookiecutterException)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError since it's an external dependency in the type hint
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format based on the class implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    
    # Act
    actual_str = str(exception)
    
    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #12
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "The input directory is not templated."
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #14
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #16
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #19
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #21
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #24
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #25
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception test"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #28
#--------------------------

```python
def test_UnknownExtension():
    message = "Error loading extension"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #32
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #33
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #34
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    context = {"project_name": "my_project", "version": "1.0.0"}

    # Execution
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )

    # Assertions for constructor attributes
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context

    # Assertions for __str__ implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #36
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #37
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #39
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #40
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #41
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exception = CookiecutterException(message)
    assert str(exception) == message
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #47
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #49
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #50
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #51
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #53
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #54
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    context = {"project_name": "my_project", "version": "1.0.0"}

    # Act
    exception = UndefinedVariableInTemplate(message, mock_error, context)

    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #56
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #57
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #58
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #59
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is required for the constructor
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    assert "Variable not found" in str(exception)
    assert "UndefinedError: 'foo' is undefined" in str(exception)
    assert "'project_name': 'my_project'" in str(exception)
```


# LLM-generated content at query #60
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #61
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #62
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Directory already exists"
    exception = OutputDirExistsException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #63
#--------------------------

```python
def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #64
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #65
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #66
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking Jinja2 TemplateError since we don't have the actual object
    mock_error = MagicMock()
    mock_error.message = "undefined symbol 'foo'"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #67
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #68
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #69
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repo type encountered"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #70
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is used in the exception
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format based on implementation:
    # "{self.message}. Error message: {self.error.message}. Context: {self.context}"
    expected_str = "Variable not found. Error message: UndefinedError: 'foo' is undefined. Context: {'project_name': 'my_project'}"
    
    # Act
    actual_str = str(exception)
    
    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #71
#--------------------------

```python
import pytest

def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #72
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #73
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #74
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #75
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML configuration"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #76
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #77
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #78
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "The input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #79
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #80
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #81
#--------------------------

```python
def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #82
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #83
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError since it's imported from jinja2 in TYPE_CHECKING
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Test attribute assignment
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test string representation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #84
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is imported from jinja2 via TYPE_CHECKING
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'missing_var' is undefined"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test __str__ implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #85
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #86
#--------------------------

```python
def test_FailedHookException():
    message = "A hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #87
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #88
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #89
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #90
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #91
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #92
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #93
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #94
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #95
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #96
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #97
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #98
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #99
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #100
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #101
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #102
#--------------------------

```python
def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #103
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #104
#--------------------------

```python
def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #105
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #106
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking jinja2.TemplateError
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'user' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Act
    result = str(exception)
    
    # Assert
    expected_output = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert result == expected_output
```


# LLM-generated content at query #107
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mock the TemplateError and its message attribute
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format based on the implementation:
    # "{self.message}. Error message: {self.error.message}. Context: {self.context}"
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    
    # Assert
    assert str(exception) == expected_str
```


# LLM-generated content at query #108
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #109
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Zip repository is invalid"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #110
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #111
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #112
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #113
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #114
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #115
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML configuration"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #116
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #117
#--------------------------

```python
import pytest

def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #118
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #119
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #120
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #121
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Mock the TemplateError object from jinja2
    mock_error = MagicMock()
    mock_error.message = "Undefined variable 'foo'"
    
    message = "A template error occurred"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Instantiate the exception
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Define the expected string format based on the class implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    
    # Assert that __str__ returns the correctly formatted string
    assert str(exception) == expected_str
```


# LLM-generated content at query #122
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #123
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type encountered"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #124
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #125
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #126
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "A hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #127
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #128
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError as it is required by the constructor
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'undefined_var' is undefined"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    assert "Variable not found" in str(exception)
    assert "UndefinedError: 'undefined_var' is undefined" in str(exception)
    assert "'project_name': 'my_project'" in str(exception)
```


# LLM-generated content at query #129
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #130
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #131
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #132
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #133
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #134
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exc = MissingProjectDir(message)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #135
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exc = CookiecutterException(message)
    assert str(exc) == message
    assert isinstance(exc, Exception)
```


# LLM-generated content at query #136
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #137
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #138
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #139
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #140
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #141
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #142
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #143
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #144
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #145
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "The provided zip file is invalid"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #146
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #147
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exception = CookiecutterException(message)
    
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #148
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #149
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #150
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #151
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exc = CookiecutterException(message)
    
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #152
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #153
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Error message"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #154
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #155
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    """
    Test that InvalidZipRepository can be instantiated and is a subclass 
    of CookiecutterException.
    """
    message = "The provided zip repository is invalid."
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #156
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #157
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Zip repository is invalid"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #158
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError as it is used in the constructor
    mock_error = MagicMock()
    mock_error.message = "Undefined symbol: {{ unknown_var }}"
    
    # Act
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Assert
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Test __str__ implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #159
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is a subclass of Exception
    # We need to mock the 'message' attribute as it's accessed in __str__
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Expected string format:
    # "{self.message}. Error message: {self.error.message}. Context: {self.context}"
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    
    # Act
    actual_str = str(exception)
    
    # Assert
    assert actual_str == expected_str
```


# LLM-generated content at query #160
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Test error message"
    exception = CookiecutterException(message)
    
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #161
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #162
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #163
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #164
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Create a mock for TemplateError as it requires a .message attribute
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Act
    result = str(exception)
    
    # Assert
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert result == expected_str
```


# LLM-generated content at query #165
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #166
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #167
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #168
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #169
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #170
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #171
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError and its message attribute
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'foo' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    expected_output = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    
    # Act
    actual_output = str(exception)
    
    # Assert
    assert actual_output == expected_output
```


# LLM-generated content at query #172
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #173
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #174
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #175
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #176
#--------------------------

```python
def test_MissingProjectDir():
    message = "Directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #177
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #178
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


