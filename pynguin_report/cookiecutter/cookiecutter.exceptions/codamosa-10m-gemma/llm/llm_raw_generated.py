####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #3
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #5
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #6
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "A hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #9
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #11
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #12
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
    assert "{'project_name': 'my_project'}" in str(exception)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Test error message"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "The provided repository is not a valid zip file."
    exception = InvalidZipRepository(message)
    
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
def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup mock TemplateError
    mock_error = MagicMock()
    mock_error.message = "Variable 'foo' is undefined"
    
    # Setup exception parameters
    message = "Template rendering failed"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Instantiate the exception
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
    
    # Assertions
    assert str(exception) == expected_str
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Setup
    message = "Variable not found"
    error_mock = MagicMock()
    error_mock.message = "jinja2 error message"
    context = {"project_name": "my_project", "version": "1.0.0"}

    # Execution
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=error_mock, 
        context=context
    )

    # Assertions
    assert exception.message == message
    assert exception.error == error_mock
    assert exception.context == context
    
    expected_str = (
        f"{message}. "
        f"Error message: {error_mock.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #22
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
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
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #26
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


# LLM-generated content at query #27
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #28
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "The input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #31
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError which is a Jinja2 exception
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


# LLM-generated content at query #33
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #34
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    error = MagicMock()
    error.message = "jinja2 error message"
    context = {"project_name": "my_project"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == f"{message}. Error message: {error.message}. Context: {context}"
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Setup mock TemplateError
    mock_error = MagicMock()
    mock_error.message = "variable 'foo' is undefined"
    
    # Setup inputs
    message = "Template error occurred"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Instantiate exception
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


# LLM-generated content at query #39
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


# LLM-generated content at query #40
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #41
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError since it's an external dependency
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
    
    # Verify __str__ implementation
    expected_str = f"{message}. Error message: {mock_error.message}. Context: {context}"
    assert str(exception) == expected_str
```


# LLM-generated content at query #43
#--------------------------

```python
def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Test error message"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #46
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #47
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    error = MagicMock()
    error.message = "jinja2 error message"
    context = {"project_name": "my_project"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == f"{message}. Error message: {error.message}. Context: {context}"
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking jinja2.TemplateError since it is used in the constructor
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
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #53
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
    
    # Expected string format based on the implementation:
    # "{message}. Error message: {error.message}. Context: {context}"
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


# LLM-generated content at query #54
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception message"
    exc = CookiecutterException(message)
    
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #56
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #57
#--------------------------

```python
def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #58
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #59
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #60
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #61
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #62
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #63
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #64
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError as it is a type hint dependency
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    assert str(exception) == f"{message}. Error message: {mock_error.message}. Context: {context}"
```


# LLM-generated content at query #65
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #66
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #67
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML configuration"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #68
#--------------------------

```python
def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #69
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #70
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #71
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #72
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #73
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #74
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #75
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #76
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Mocking TemplateError since it's a type hint dependency
    mock_error = MagicMock()
    mock_error.message = "undefined variable 'foo'"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    assert "Variable not found" in str(exception)
    assert "undefined variable 'foo'" in str(exception)
    assert "'project_name': 'my_project'" in str(exception)
```


# LLM-generated content at query #77
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #78
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #79
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #80
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exc = MissingProjectDir(message)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #81
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Mock the TemplateError which is used in the exception
    mock_error = MagicMock()
    mock_error.message = "variable 'foo' is undefined"
    
    message = "An error occurred"
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


# LLM-generated content at query #82
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #83
#--------------------------

```python
def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #84
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #85
#--------------------------

```python
def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #2
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Test error message"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #5
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #6
#--------------------------

```python
def test_UnknownExtension():
    message = "Test error message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "Invalid zip repository"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "The provided repository is not a valid zip archive"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #10
#--------------------------

```python
def test_UnknownExtension():
    message = "Extension not found"
    exception = UnknownExtension(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #13
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #14
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    error = MagicMock()
    error.message = "UndefinedError: 'my_var' is undefined"
    context = {"project_name": "test_project"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == f"{message}. Error message: {error.message}. Context: {context}"
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'my_var' is undefined"
    context = {"project_name": "test_project", "version": "1.0.0"}

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


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #19
#--------------------------

```python
def test_UnknownExtension():
    message = "An error occurred"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Create a mock for TemplateError and its message attribute
    mock_error = MagicMock()
    mock_error.message = "UndefinedError: 'missing_var' is undefined"
    
    exception = UndefinedVariableInTemplate(
        message=message, 
        error=mock_error, 
        context=context
    )
    
    # Verify attributes are correctly assigned
    assert exception.message == message
    assert exception.error == mock_error
    assert exception.context == context
    
    # Verify the string representation matches the implementation
    expected_str = (
        f"{message}. "
        f"Error message: {mock_error.message}. "
        f"Context: {context}"
    )
    assert str(exception) == expected_str
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #22
#--------------------------

```python
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

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #26
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #28
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #29
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception message"
    exception = CookiecutterException(message)
    
    assert isinstance(exception, Exception)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate___str__():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project", "version": "1.0.0"}
    
    # Mocking TemplateError from jinja2
    mock_error = MagicMock()
    mock_error.message = "jinja2 error message"
    
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


# LLM-generated content at query #32
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #35
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #36
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest

def test_OutputDirExistsException():
    message = "The output directory already exists."
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #40
#--------------------------

```python
def test_InvalidZipRepository():
    message = "The provided zip repository is invalid."
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #41
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_CookiecutterException():
    message = "Base exception message"
    exc = CookiecutterException(message)
    
    assert isinstance(exc, Exception)
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #43
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #45
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_RepositoryCloneFailed():
    message = "Failed to clone repository"
    exception = RepositoryCloneFailed(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #47
#--------------------------

```python
def test_InvalidZipRepository():
    message = "Test error message"
    exception = InvalidZipRepository(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #48
#--------------------------

```python
def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes: no_input and replay cannot be used together."
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #50
#--------------------------

```python
def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #51
#--------------------------

```python
def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #53
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #54
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
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
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #58
#--------------------------

```python
import pytest

def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #59
#--------------------------

```python
def test_InvalidModeException():
    message = "Incompatible modes provided"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #60
#--------------------------

```python
import pytest

def test_UnknownRepoType():
    message = "Unknown repository type"
    exception = UnknownRepoType(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #61
#--------------------------

```python
def test_FailedHookException():
    message = "A hook failed"
    exception = FailedHookException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #62
#--------------------------

```python
import pytest

def test_InvalidConfiguration():
    message = "Invalid YAML format"
    exception = InvalidConfiguration(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #65
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name is empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #66
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #67
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "The input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
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

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #70
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #71
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exc = MissingProjectDir(message)
    
    assert isinstance(exc, CookiecutterException)
    assert str(exc) == message
```


# LLM-generated content at query #72
#--------------------------

```python
import pytest

def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #73
#--------------------------

```python
import pytest

def test_FailedHookException():
    message = "Hook execution failed"
    exception = FailedHookException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #74
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    # Arrange
    message = "Variable not found"
    context = {"project_name": "my_project"}
    
    # Create a mock for TemplateError which is expected to have a .message attribute
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


# LLM-generated content at query #75
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exc = VCSNotInstalled(message)
    assert str(exc) == message
    assert isinstance(exc, CookiecutterException)
```


# LLM-generated content at query #76
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #77
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #78
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repository type encountered"
    exception = UnknownRepoType(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #79
#--------------------------

```python
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

def test_InvalidZipRepository():
    message = "The provided zip file is not a valid archive"
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #81
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #82
#--------------------------

```python
import pytest

def test_UnknownExtension():
    message = "Failed to import extension"
    exception = UnknownExtension(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #83
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #84
#--------------------------

```python
import pytest

def test_ContextDecodingException():
    message = "Failed to decode JSON context"
    exception = ContextDecodingException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #85
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes detected"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #86
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #87
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #88
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #89
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #90
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #91
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #92
#--------------------------

```python
import pytest

def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #93
#--------------------------

```python
def test_RepositoryNotFound():
    message = "Repository not found"
    exception = RepositoryNotFound(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #94
#--------------------------

```python
import pytest

def test_NonTemplatedInputDirException():
    message = "Input directory is not templated"
    exception = NonTemplatedInputDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #95
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_UndefinedVariableInTemplate():
    message = "Variable not found"
    error = MagicMock()
    error.message = "jinja2 error message"
    context = {"project_name": "my_project"}

    exception = UndefinedVariableInTemplate(message, error, context)

    assert exception.message == message
    assert exception.error == error
    assert exception.context == context
    assert str(exception) == f"{message}. Error message: {error.message}. Context: {context}"
```


# LLM-generated content at query #96
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #97
#--------------------------

```python
def test_UnknownRepoType():
    message = "Unknown repo type"
    exception = UnknownRepoType(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #98
#--------------------------

```python
import pytest

def test_InvalidModeException():
    message = "Incompatible modes provided"
    exception = InvalidModeException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #99
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory."
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #100
#--------------------------

```python
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

def test_UnknownTemplateDirException():
    message = "Ambiguous template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #102
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #103
#--------------------------

```python
def test_VCSNotInstalled():
    message = "Git is not installed"
    exception = VCSNotInstalled(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #104
#--------------------------

```python
def test_ConfigDoesNotExistException():
    message = "Config file not found"
    exception = ConfigDoesNotExistException(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #105
#--------------------------

```python
def test_OutputDirExistsException():
    message = "Output directory already exists"
    exception = OutputDirExistsException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #106
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #107
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid configuration file"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #108
#--------------------------

```python
def test_MissingProjectDir():
    message = "Project directory not found"
    exception = MissingProjectDir(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #109
#--------------------------

```python
def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
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
def test_InvalidConfiguration():
    message = "Invalid YAML syntax"
    exception = InvalidConfiguration(message)
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #112
#--------------------------

```python
def test_UnknownTemplateDirException():
    message = "Ambiguous project template directory"
    exception = UnknownTemplateDirException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #113
#--------------------------

```python
def test_UnknownExtension():
    message = "Test message"
    exception = UnknownExtension(message)
    assert str(exception) == message
    assert isinstance(exception, CookiecutterException)
```


# LLM-generated content at query #114
#--------------------------

```python
import pytest

def test_InvalidZipRepository():
    message = "The provided repository is not a valid zip file."
    exception = InvalidZipRepository(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


# LLM-generated content at query #115
#--------------------------

```python
import pytest

def test_EmptyDirNameException():
    message = "Directory name cannot be empty"
    exception = EmptyDirNameException(message)
    
    assert isinstance(exception, CookiecutterException)
    assert str(exception) == message
```


