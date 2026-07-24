####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Ensure working directory is restored after using work_in with a directory
    initial_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == initial_dir

    # Test 2: Ensure working directory remains the same when work_in is called with None
    initial_dir_2 = os.getcwd()
    with work_in(None):
        assert os.getcwd() == initial_dir_2
    
    assert os.getcwd() == initial_dir_2

    # Test 3: Ensure working directory is restored even if an exception occurs inside the block
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with pytest.raises(RuntimeError):
            with work_in(tmp_path):
                raise RuntimeError("Simulated error")
        
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test case 1: Using the context manager without arguments (should stay in original dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test case 2: Changing to a new directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test case 3: Verifying we return to original directory after exit
        assert os.getcwd() == original_cwd

    # Final sanity check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_simple_filter():
    # Define a dummy filter function
    def dummy_filter(value):
        return value.upper()

    # Apply the decorator
    ExtensionClass = simple_filter(dummy_filter)

    # Create a mock Jinja Environment
    mock_env = MagicMock()
    mock_env.filters = {}

    # Instantiate the extension class with the mock environment
    extension_instance = ExtensionClass(mock_env)

    # Verify the extension class name matches the function name
    assert ExtensionClass.__name__ == "dummy_filter"

    # Verify the filter was actually added to the environment's filters dictionary
    assert "dummy_filter" in mock_env.filters
    assert mock_env.filters["dummy_filter"] == dummy_filter

    # Verify the logic of the filter itself works as expected
    assert mock_env.filters["dummy_filter"]("hello") == "HELLO"
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: Get current working directory and create a temp directory
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test 1: Context manager without arguments (should not change directory)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Context manager with directory (should change directory)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Verify we return to original directory after context exits
        assert os.getcwd() == original_cwd

    # Final check to ensure no side effects leaked
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        original_cwd = os.getcwd()
        
        # Create a sub-directory to work in
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        # Test Case 1: Using work_in with a directory
        with work_in(subdir):
            assert os.getcwd() == str(subdir)
            # Verify we can still perform operations in the new dir
            test_file = subdir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Test Case 2: Verify we returned to the original directory
        assert os.getcwd() == original_cwd
        
        # Test Case 3: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test Case 4: Verify directory switching works with string paths
        with work_in(str(tmp_path)):
            assert os.getcwd() == str(tmp_path)
            
        # Final check: Ensure we are back to original
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test the work_in context manager."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test 1: Without dirname (should stay in original directory)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: With dirname (should change to tmp_path)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: After exiting context (should return to original directory)
        assert os.getcwd() == original_cwd

    # Final check to ensure no side effects left the process in a different directory
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes the working directory and returns to the original."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test with a specific directory
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
        
        # Test that we returned to the original directory
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

        # Test with None (should stay in current directory)
        with work_in(None):
            assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Final safety check
    assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory remains unchanged when no dirname is provided
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Verify working directory changes and reverts when dirname is provided
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_cwd

    # Test 3: Verify working directory changes and reverts using a nested path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        subdir_path = tmp_path / "subdir"
        subdir_path.mkdir()
        
        with work_in(subdir_path):
            assert os.getcwd() == str(subdir_path)
        
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: Create a temporary directory to act as the target directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir).resolve()
        original_dir = Path(os.getcwd()).resolve()

        # Test 1: Ensure we are in the original directory before entering context
        assert Path(os.getcwd()).resolve() == original_dir

        # Test 2: Enter the context manager and check if directory changed
        with work_in(target_dir):
            assert Path(os.getcwd()).resolve() == target_dir
            
            # Test 3: Check if we can perform operations in the new directory
            test_file = target_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()

        # Test 4: Ensure we are returned to the original directory after exiting
        assert Path(os.getcwd()).resolve() == original_dir

        # Test 5: Ensure work_in without arguments does not change the directory
        with work_in(None):
            assert Path(os.getcwd()).resolve() == original_dir
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test context manager with a new directory
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
            
        # Verify we returned to the original directory
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
        
        # Test context manager with None (should stay in current dir)
        with work_in(None):
            assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Final safety check to ensure no side effects left the test hanging
    assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
```


# LLM-generated content at query #11
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the current working directory to verify return
    original_cwd = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test entering a new directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Test that we returned to the original directory after the context manager exits
        assert os.getcwd() == original_cwd

    # Test behavior when no dirname is provided (should remain in original dir)
    with work_in(None):
        assert os.getcwd() == original_cwd

    # Test behavior with a non-existent directory (should raise error)
    non_existent_dir = Path(tempfile.mkdtemp()) / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass
```


# LLM-generated content at query #12
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return
    original_dir = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test context manager with directory change
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
            
        # Verify we returned to the original directory
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
        
        # Test context manager without directory change (None)
        with work_in(None):
            assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Final check to ensure environment is clean even if test fails mid-way
    # (Though pytest handles this, it's good practice for testing side effects)
    assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_test_filter)

    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension
    extension = ExtensionClass(env)

    # Assertions
    assert ExtensionClass.__name__ == "my_test_filter"
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    assert env.filters["my_test_filter"]("hello") == "HELLO"
```


# LLM-generated content at query #14
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return
    original_cwd = os.getcwd()
    
    # Create a temporary directory to work in
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test 1: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Verify we return to original directory after context exit
        assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify restoration
    original_cwd = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test case 1: Using work_in without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test case 2: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            # Verify we can perform operations in the new directory
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Test case 3: Verify restoration after exiting context
        assert os.getcwd() == original_cwd

    # Test case 4: Using a string path instead of Path object
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        with work_in(str(Path(tmp_dir_str))):
            assert os.getcwd() == str(Path(tmp_dir_str))
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test changing to a new directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Test returning to the original directory
        assert os.getcwd() == original_dir
        
        # Test work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_dir

    # Ensure we are back to original dir even if error occurs inside context
    assert os.getcwd() == original_dir

def test_work_in_error_handling():
    """Test that work_in restores directory even if an exception is raised."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        with pytest.raises(RuntimeError):
            with work_in(tmp_path):
                raise RuntimeError("Simulated error")
        
        # Verify restoration after exception
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a sample filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Create a Jinja2 environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension = ExtensionClass(env)

    # Verify the extension was registered in the environment's filters
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify the functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"

    # Verify the class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Use the decorator/factory to create the extension class
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Create a Jinja2 environment
    env = Environment()
    
    # Instantiate the extension with the environment
    # In Jinja2, extensions are typically added via the environment
    extension_instance = ExtensionClass(env)
    
    # Assert the extension name was correctly assigned
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Assert the filter was actually registered in the environment
    assert "my_custom_filter" in env.filters
    
    # Test the functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #19
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the current working directory to verify return
    original_cwd = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test 1: Moving into a directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify we returned to original directory after context exit
        assert os.getcwd() == original_cwd
        
        # Test 2: Using work_in without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #20
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: Get original directory and create a temporary directory
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Create a file inside the temp directory to verify we are actually there
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("hello")

        # Case 1: Test working in a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            assert os.path.exists("test_file.txt")
        
        # Verify we returned to original directory after context exit
        assert os.getcwd() == original_dir

        # Case 2: Test working in None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

        # Case 3: Test working in a subdirectory of the temp directory
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        with work_in(sub_dir):
            assert os.getcwd() == str(sub_dir)
        
        # Verify we returned to original directory after nested exit
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test the work_in context manager."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test 1: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Ensure we return to original directory after context exit
        assert os.getcwd() == original_cwd

    # Final check to ensure no side effects leaked
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Get the decorated extension class
    ExtensionClass = simple_filter(my_test_filter)

    # Create a Jinja environment
    env = Environment()

    # Instantiate the extension with the environment
    extension = ExtensionClass(env)

    # Verify the extension name matches the function name
    assert ExtensionClass.__name__ == "my_test_filter"

    # Verify the filter was actually added to the environment
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter

    # Verify the filter functionality works within the environment
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension = ExtensionClass(env)

    # Verify the filter was added to the environment
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify the functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"

    # Verify the extension class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #24
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the current working directory to verify return
    original_cwd = os.getcwd()
    
    # Create a temporary directory to switch into
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test context manager without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test context manager with a directory argument
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Verify that after exiting the context manager, we are back to original
        assert os.getcwd() == original_cwd

    # Double check final state
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #25
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes and restores the working directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test 1: Context manager without dirname (should not change directory)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Context manager with dirname (should change directory)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Verify restoration after exiting context
        assert os.getcwd() == original_cwd

    # Ensure we are back to original even if something went wrong in the test logic
    os.chdir(original_cwd)
```


# LLM-generated content at query #26
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in changes the directory and restores it on exit."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with dirname provided
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Test that it returns to original directory
        assert os.getcwd() == original_cwd
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_cwd

    # Final check to ensure no side effects leaked
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #27
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the current working directory to verify we return to it
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test context manager with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify we returned to the original directory
        assert os.getcwd() == original_cwd

    # Test context manager without arguments (should stay in same dir)
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #28
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the current working directory to verify return
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify we returned to the original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    # Final check to ensure no side effects left the process in tmp_dir
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #29
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify it returns to the original directory after exiting context
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir

    # Test 2: Verify behavior when no directory is provided (should stay in current dir)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test 3: Verify error handling (should return to original dir even if exception occurs)
    with pytest.raises(RuntimeError):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            with work_in(tmp_path):
                raise RuntimeError("Simulated error")
    
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator-like function
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension
    extension = ExtensionClass(env)

    # Verify the extension name was correctly assigned
    assert ExtensionClass.__name__ == "my_custom_filter"

    # Verify the filter is now available in the environment
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify the filter actually works within the environment context
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #31
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Using work_in without arguments (should not change directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Using work_in with a directory (should change and revert directory)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the original directory
        assert os.getcwd() == original_dir

    # Test 3: Verifying behavior with nested work_in calls
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        with work_in(tmp_path):
            with work_in(tmp_path):
                assert os.getcwd() == tmp_path
            assert os.getcwd() == tmp_path
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Get the extension class created by the decorator
    ExtensionClass = simple_filter(my_custom_filter)

    # Create a Jinja Environment
    env = Environment()

    # Initialize the extension with the environment
    # The decorator logic registers the filter during __init__
    extension = ExtensionClass(env)

    # Assertions
    assert my_custom_filter.__name__ in env.filters
    assert env.filters[my_custom_filter.__name__] == my_custom_filter
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter actually works in the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #33
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get initial working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test context manager with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Test that it returns to the original directory after exit
        assert os.getcwd() == original_dir
        
        # Test context manager without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_dir

    # Final sanity check to ensure we are back where we started
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #34
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test the work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with dirname provided
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify directory was restored after context exit
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    # Final check to ensure no side effects leaked
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #35
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Create a temporary directory to work in
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        original_cwd = os.getcwd()
        
        # Test 1: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Using work_in with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Verify we return to the original directory after context exit
        assert os.getcwd() == original_cwd

        # Test 4: Verify functionality inside a nested work_in
        with work_in(tmp_path):
            with work_in(tmp_path / "subdir"):
                # Create subdir for the test
                os.makedirs(tmp_path / "subdir", exist_ok=True)
                assert os.getcwd() == tmp_path / "subdir"
            assert os.getcwd() == tmp_path
```


# LLM-generated content at query #36
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify we return to it
    original_cwd = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test case 1: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Verify we returned to the original directory after context exit
        assert os.getcwd() == original_cwd
        
        # Test case 2: Using work_in without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd

    # Final sanity check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #37
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Store the original working directory to verify we return to it
    original_cwd = os.getcwd()
    
    # Create a temporary directory to use as the new working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Verify we are not in the tmp_path initially
        assert os.getcwd() != str(tmp_path)
        
        # Test context manager with a directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
            # Create a file inside the tmp directory to verify we can interact with it
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Verify we have returned to the original working directory
        assert os.getcwd() == original_cwd

    # Test context manager with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def mock_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    extension_class = simple_filter(mock_filter)
    
    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension
    extension_instance = extension_class(env)
    
    # Verify the extension was registered in the environment's filters
    assert "mock_filter" in env.filters
    assert env.filters["mock_filter"] == mock_filter
    
    # Verify the functionality of the registered filter
    assert env.from_string("hello").render() == "HELLO"
    
    # Verify the class name matches the function name
    assert extension_class.__name__ == "mock_filter"
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def mock_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(mock_filter)

    # Create a Jinja environment
    env = Environment()
    
    # Initialize the extension with the environment
    extension = ExtensionClass(env)

    # Verify the extension was correctly registered as a filter
    assert mock_filter.__name__ in env.filters
    assert env.filters[mock_filter.__name__] == mock_filter

    # Verify the filter actually works in the environment
    template = env.from_string("{{ 'hello' | upper_case_test }}")
    # Note: simple_filter uses the function's __name__ as the filter name.
    # Since we named the function 'mock_filter', the filter name is 'mock_filter'
    
    # Let's re-test with a specific name check
    def test_func(x):
        return x
    
    Ext = simple_filter(test_func)
    env_test = Environment()
    Ext(env_test)
    
    assert "test_func" in env_test.filters
    assert env_test.filters["test_func"](1) == 1
    
    # Verify the class name was updated to the function name
    assert Ext.__name__ == "test_func"
```


# LLM-generated content at query #40
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes and restores the working directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with no argument (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test with directory argument (should change dir)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test that it reverts after the context manager exits
        assert os.getcwd() == original_cwd

    # Final check to ensure the environment is clean
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #41
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension
    extension = ExtensionClass(env)
    
    # Verify the extension name was correctly assigned
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter was actually added to the environment
    assert "my_custom_filter" in env.filters
    
    # Verify the filter functionality works within the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #42
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get original working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory to work in
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        # Verify we are not in the tmp directory initially
        assert os.getcwd() == original_cwd
        
        # Test context manager with a new directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            # Verify we can perform operations in the new dir
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Verify we returned to the original directory
        assert os.getcwd() == original_cwd

    # Test context manager with None (should stay in current dir)
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #43
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get the original working directory
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test 1: Using work_in without arguments (should stay in original dir)
        with work_in():
            assert os.getcwd() == original_dir
            
        # Test 2: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test 3: Verify we return to original directory after context exit
        assert os.getcwd() == original_dir

    # Test 4: Test with a nested directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        nested_path = tmp_path / "nested"
        nested_path.mkdir()
        
        with work_in(nested_path):
            assert os.getcwd() == nested_path
            
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #44
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: Get current directory to verify we return to it
    original_dir = os.getcwd()
    
    # Create a temporary directory to move into
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Test 1: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            # Verify we can perform operations in the new directory
            new_file = tmp_path / "test_file.txt"
            new_file.write_text("hello")
            assert new_file.exists()
            
        # Verify we returned to the original directory
        assert os.getcwd() == original_dir

        # Test 2: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_dir

    # Final safety check
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #45
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return to original
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test changing to a new directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Test returning to original directory after context exit
        assert os.getcwd() == original_cwd
        
        # Test context manager with None (should stay in current dir)
        with work_in(None):
            assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #46
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Setup: Get current directory and create a temporary directory
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Create a sub-directory inside the temp directory
        sub_dir = tmp_path / "sub_folder"
        sub_dir.mkdir()

        # Test 1: Using work_in with a directory name
        with work_in(sub_dir):
            assert os.getcwd() == str(sub_dir)
        
        # Verify we returned to the original directory
        assert os.getcwd() == original_dir

        # Test 2: Using work_in without any arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_dir

        # Test 3: Using work_in with a non-existent directory (should raise error)
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            with work_in(non_existent):
                pass
        
        # Verify we are still in the original directory after the failure
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #47
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Create a Jinja environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension = ExtensionClass(env)
    
    # Verify the filter was added to the environment
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    # Verify the functionality of the filter through the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
    
    # Verify the class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #48
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test case 1: Using work_in with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify we returned to the original directory
        assert os.getcwd() == original_dir
        
        # Test case 2: Using work_in without arguments (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_dir

    # Final sanity check
    assert os.getcwd() == original_dir
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    new_dir = tmp_path / "test_subdir"
    new_dir.mkdir()

    # Test 1: Context manager without dirname (should stay in original directory)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test 2: Context manager with dirname (should change and revert)
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    assert os.getcwd() == original_cwd

    # Test 3: Ensure it handles nested changes correctly
    sub_subdir = new_dir / "inner"
    sub_subdir.mkdir()
    with work_in(new_dir):
        with work_in(sub_subdir):
            assert os.getcwd() == str(sub_subdir.resolve())
        assert os.getcwd() == str(new_dir.resolve())

    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test working directory remains unchanged when no argument is passed
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test working directory changes to the specified path and reverts back
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir

    # Test working directory changes using a relative path
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        # Create a subdirectory to move into
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(subdir):
            assert os.getcwd() == str(subdir)
            
        assert os.getcwd() == original_dir

    # Test working directory changes using a relative path string
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        # Create a file inside the temp dir to use as reference
        ref_file = tmp_path / "test.txt"
        ref_file.write_text("hello")
        
        # Use relative path from current working directory to the tmp_path
        # We'll find the relative path between original_dir and tmp_path
        rel_path = os.path.relpath(tmp_path, original_dir)
        
        with work_in(rel_path):
            assert os.getcwd() == str(tmp_path)
            
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Create a Jinja Environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    # This should register 'my_custom_filter' in env.filters
    extension = ExtensionClass(env)
    
    # Verify the function name is correctly assigned to the class
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter was actually added to the environment
    assert "my_custom_filter" in env.filters
    
    # Verify the logic of the filter works through the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"

def test_simple_filter_multiple():
    def filter_one(x): return x + 1
    def filter_two(x): return x * 2

    env = Environment()
    
    # Test multiple independent registrations
    Ext1 = simple_filter(filter_one)
    Ext2 = simple_filter(filter_two)
    
    Ext1(env)
    Ext2(env)
    
    assert env.filters["filter_one"] == filter_one
    assert env.filters["filter_two"] == filter_two
    assert env.from_string("{{ 1 | filter_one | filter_two }}") == 4
```


# LLM-generated content at query #4
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    
    # Test context manager without arguments (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_cwd
        
    # Create a temporary directory to move into
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()
    
    # Test context manager with directory change
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    # Ensure we returned to original directory after exit
    assert os.getcwd() == original_cwd

def test_work_in_with_non_existent_dir():
    original_cwd = os.getcwd()
    non_existent = Path("/tmp/this/path/should/not/exist/at/all")
    
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
            
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #5
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a dummy directory to change into
    subdir = tmp_path / "test_dir"
    subdir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test case 1: Changing directory via context manager
    with work_in(subdir):
        assert os.getcwd() == str(subdir.resolve())
    
    # Ensure we returned to the original directory after exiting context
    assert os.getcwd() == original_cwd

    # Test case 2: Using context manager without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test case 3: Working with a non-existent directory should raise error, 
    # but the finally block must still restore the original CWD
    try:
        non_existent = tmp_path / "does_not_exist"
        with work_in(non_existent):
            pass
    except FileNotFoundError:
        pass
    
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #6
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test with dirname provided
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    # Test that it returns to original directory
    assert os.getcwd() == original_cwd

    # Test without dirname (should stay in current dir)
    with work_in(None):
        assert os.getcwd() == original_cwd

    # Test with a non-existent directory (should raise FileNotFoundError via os.chdir)
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
```


# LLM-generated content at query #7
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    """Test that work_in changes directory and returns to original."""
    original_dir = os.getcwd()
    new_dir = tmp_path / "sub_dir"
    new_dir.mkdir()

    # Case 1: No argument provided (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Case 2: Argument provided (should change and return)
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    assert os.getcwd() == original_dir

    # Case 3: Error within context (should still return to original)
    try:
        with work_in(new_dir):
            raise ValueError("Simulated error")
    except ValueError:
        pass
    
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja2 Environment
    env = Environment()
    
    # Instantiate the generated extension class
    extension = ExtensionClass(env)

    # Verify that the filter was registered in the environment
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify functionality of the registered filter
    template = env.from_string("hello {{ name|my_custom_filter }}")
    rendered = template.render(name="world")
    assert rendered == "hello WORLD"

    # Verify that the class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without dirname (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the original directory after exit
        assert os.getcwd() == original_dir

    # Test 3: Verifying it works with string paths as well as Path objects
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        with work_in(tmp_dir_str):
            assert os.path.abspath(os.getcwd()) == os.path.abspath(tmp_dir_str)
        assert os.path.abspath(os.getcwd()) == os.path.abspath(original_dir)
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the original directory after exiting context
        assert os.getcwd() == original_cwd

    # Test 3: Context manager with a subdirectory
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir).resolve()
        sub_path = base_path / "subdir"
        sub_path.mkdir()
        
        assert os.getcwd() == original_cwd
        with work_in(sub_path):
            assert os.getcwd() == sub_path
        
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #11
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test case 1: Using context manager with a directory
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
        
    # Verify we returned to original directory
    assert os.getcwd() == original_cwd

    # Test case 2: Using context manager without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test case 3: Ensure it handles path strings as well as Path objects
    with work_in(str(new_dir)):
        assert os.getcwd() == str(new_dir.resolve())
```


# LLM-generated content at query #12
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Test with no arguments (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir

    # Test with a specific directory
    test_dir = tmp_path / "subfolder"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir.resolve())
    
    # Test that it returns to the initial directory after context exit
    assert os.getcwd() == initial_dir

    # Test with a nested directory
    nested_dir = test_dir / "nested"
    nested_dir.mkdir()
    
    with work_in(nested_dir):
        assert os.getcwd() == str(nested_dir.resolve())
    
    assert os.getcwd() == initial_dir
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: Using work_in without arguments (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir

    # Test case 2: Using work_in with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the original directory after context exit
        assert os.getcwd() == initial_dir

    # Test case 3: Verifying nested usage
    with tempfile.TemporaryDirectory() as tmp_dir_1:
        tmp_path_1 = Path(tmp_dir_1).resolve()
        with tempfile.TemporaryDirectory() as tmp_dir_2:
            tmp_path_2 = Path(tmp_dir_2).resolve()
            
            with work_in(tmp_path_1):
                assert os.getcwd() == str(tmp_path_1)
                with work_in(tmp_path_2):
                    assert os.getcwd() == str(tmp_path_2)
                assert os.getcwd() == str(tmp_path_1)
            
            assert os.getcwd() == initial_dir
```


# LLM-generated content at query #14
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
        # After exiting, should return to original directory
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Test 3: Context manager with a non-existent directory (should raise FileNotFoundError)
    non_existent_dir = Path(tempfile.mkdtemp()) / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass
    # Ensure we still returned to original directory even after exception
    assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in same directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir
    assert os.getcwd() == initial_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the original directory
        assert os.getcwd() == initial_dir

    # Test 3: Verify it works with a subdirectory within the temp dir
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        
        with work_in(sub_dir):
            assert os.getcwd() == str(sub_dir)
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: No directory provided (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test case 2: Directory provided (should change and return)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify we returned to original directory after context exit
        assert os.getcwd() == original_dir

    # Test case 3: Nested directory changes
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #17
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test behavior without argument (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir
    assert os.getcwd() == initial_dir

    # Test behavior with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Verify return to original directory after context exit
        assert os.getcwd() == initial_dir

    # Test behavior with a nested directory structure
    with tempfile.TemporaryDirectory() as base_tmp:
        nested_tmp = Path(base_tmp) / "subdir" / "deep"
        nested_tmp.mkdir(parents=True)
        
        with work_in(nested_tmp):
            assert os.getcwd() == str(nested_tmp.resolve())
        
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #18
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test context manager without argument (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_cwd
        
    # Test context manager with directory argument
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        
    # Ensure we returned to the original directory after exit
    assert os.getcwd() == original_cwd

    # Test with a non-existent directory (should raise FileNotFoundError on chdir)
    non_existent = tmp_path / "ghost"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
```


# LLM-generated content at query #19
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify current directory is preserved when no dirname is provided
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Verify changing to a new directory and returning to original
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # After exiting context, must return to original
        assert os.getcwd() == original_dir

    # Test 3: Verify behavior with nested context managers
    with tempfile.TemporaryDirectory() as inner_dir:
        inner_path = Path(inner_dir).resolve()
        with work_in(inner_path):
            assert os.getcwd() == inner_path
            with work_in(original_dir):
                assert os.getcwd() == original_dir
            assert os.getcwd() == inner_path
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Use the decorator/wrapper
    ExtensionClass = simple_filter(my_test_filter)
    
    # Create a Jinja environment
    env = Environment()
    
    # Instantiate the extension class within the environment
    extension = ExtensionClass(env)
    
    # Verify that the filter was added to the environment
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    
    # Verify functionality of the registered filter
    assert env.from_string("hello").render() == "HELLO"

    # Verify that the extension class name is correctly set to the function name
    assert ExtensionClass.__name__ == "my_test_filter"
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test original working directory is preserved
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Case 1: Using work_in without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Case 2: Using work_in with a new directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # After exiting the context manager, we must be back in original dir
        assert os.getcwd() == original_cwd

    # Verify integrity of environment after multiple jumps
    with tempfile.TemporaryDirectory() as tmpdir2:
        tmp_path2 = Path(tmpdir2).resolve()
        with work_in(tmp_path2):
            with work_in(tmp_path):
                assert os.getcwd() == tmp_path
            assert os.getcwd() == tmp_path2
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #22
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_dir = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test context manager with dirname
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())

    # Verify it returns to original directory
    assert os.getcwd() == original_dir

    # Test context manager without dirname (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test context manager with a non-existent directory 
    # (os.chdir raises FileNotFoundError, but finally block should still restore dir)
    non_existent = tmp_path / "ghost"
    try:
        with work_in(non_existent):
            pass
    except FileNotFoundError:
        pass
    
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Wrap it using simple_filter
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension (which registers the filter)
    extension = ExtensionClass(env)
    
    # Assertions
    assert my_custom_filter.__name__ in env.filters
    assert env.filters[my_custom_filter.__name__] == my_custom_filter
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter actually works in the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #24
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the initial directory after exiting context
        assert os.getcwd() == initial_dir

    # Test 3: Context manager with a nested directory
    with tempfile.TemporaryDirectory() as base_tmp:
        base_path = Path(base_tmp).resolve()
        sub_path = base_path / "subdir"
        sub_path.mkdir()
        
        with work_in(base_path):
            assert os.getcwd() == base_path
            with work_in(sub_path):
                assert os.getcwd() == sub_path
            assert os.getcwd() == base_path
        
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Use the decorator/factory to create the extension class
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize a Jinja2 environment
    env = Environment()

    # Instantiate the extension with the environment
    # In jinja2, extensions are initialized by passing the environment
    extension_instance = ExtensionClass(env)

    # Verify that the filter has been added to the environment's filters dictionary
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify functionality of the registered filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"

    # Verify the class name was correctly updated to match the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory remains unchanged when no argument is provided
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Verify working directory changes and reverts when dirname is provided
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Verify it reverted to original
        assert os.getcwd() == original_cwd

    # Test 3: Verify working directory changes and reverts when using a nested path
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir).resolve()
        nested_path = base_path / "subdir"
        nested_path.mkdir()
        
        with work_in(nested_path):
            assert os.getcwd() == str(nested_path)
        
        assert os.getcwd() == original_cwd

    # Test 4: Verify error handling (if chdir fails, it should still revert)
    with pytest.raises(OSError):
        with work_in("/non/existent/path/that/should/fail"):
            pass
    
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Create a Jinja environment
    env = Environment()
    
    # Instantiate the extension and add to environment
    extension = ExtensionClass(env)
    
    # Assertions
    assert ExtensionClass.__name__ == "my_custom_filter"
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"]( "hello") == "HELLO"

    # Test with another function to ensure no leakage or static naming issues
    def another_func(x):
        return x + 1
    
    AnotherExtensionClass = simple_filter(another_func)
    another_ext = AnotherExtensionClass(env)
    
    assert AnotherExtensionClass.__name__ == "another_func"
    assert "another_func" in env.filters
    assert env.filters["another_func"](5) == 6
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a sample filter function
    def my_uppercase_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_uppercase_filter)
    
    # Create a Jinja environment
    env = Environment()
    
    # Instantiate the extension (which registers the filter in __init__)
    ExtensionClass(env)

    # Assertions
    assert "my_uppercase_filter" in env.filters
    assert env.filters["my_uppercase_filter"] == my_uppercase_filter
    assert ExtensionClass.__name__ == "my_uppercase_filter"
    
    # Verify functionality within the environment
    template = env.from_string("{{ 'hello' | my_uppercase_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #4
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    subdir = tmp_path / "subfolder"
    subdir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test context manager with a directory
    with work_in(subdir):
        assert os.getcwd() == str(subdir.resolve())
        
    # Verify we returned to the original directory
    assert os.getcwd() == original_cwd

    # Test context manager without arguments (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test changing back and forth
    with work_in(tmp_path):
        assert os.getcwd() == str(tmp_path.resolve())
        with work_in(subdir):
            assert os.getcwd() == str(subdir.resolve())
        assert os.getcwd() == str(tmp_path.resolve())

    # Verify final state
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: Using work_in without arguments (should stay in same directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir

    # Test case 2: Using work_in with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the initial directory after context exit
        assert os.getcwd() == initial_dir

    # Test case 3: Using work_in with a nested directory structure
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        nested_path = root_path / "subdir" / "deeply"
        
        with work_in(nested_path):
            assert os.getcwd() == str(nested_path)
            # Verify we can actually perform operations in the new dir
            test_file = nested_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        assert os.getcwd() == initial_dir

    # Test case 4: Ensuring directory changes are reverted even if an exception occurs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        try:
            with work_in(tmp_path):
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
            
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Should return to original directory after exit
        assert os.getcwd() == original_cwd

    # Test 3: Context manager with a subdirectory
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()
        sub_path = root_path / "subdir"
        sub_path.mkdir()
        
        with work_in(sub_path):
            assert os.getcwd() == str(sub_path)
        
        assert os.getcwd() == original_cwd

    # Test 4: Verifying it handles string paths as well as Path objects
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_str_path = str(Path(tmpdir).resolve())
        
        with work_in(tmp_str_path):
            assert os.getcwd() == tmp_str_path
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a sample filter function
    def my_custom_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension = ExtensionClass(env)

    # Verify that the filter was correctly added to the environment's filters dictionary
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify functionality of the filter within the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"

    # Verify the extension class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a sample filter function
    def my_custom_filter(s):
        return s.upper()

    # Use the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension_instance = ExtensionClass(env)
    
    # Assertions
    assert my_custom_filter.__name__ in env.filters
    assert env.filters[my_custom_filter.__name__] == my_custom_filter
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter actually works within the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #9
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()

    # Test 1: Using work_in with a directory
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
    
    assert os.getcwd() == original_cwd

    # Test 2: Using work_in without arguments (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test 3: Ensure it handles Path objects correctly
    with work_in(Path(sub_dir)):
        assert os.getcwd() == str(sub_dir.resolve())
    
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without dirname (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir
    assert os.getcwd() == initial_dir

    # Test 2: Context manager with dirname (should change and return)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        assert os.getcwd() == initial_dir

    # Test 3: Verifying it works with nested directory structures
    with tempfile.TemporaryDirectory() as root_tmp:
        nested_tmp = Path(root_tmp) / "subdir" / "deepdir"
        nested_tmp.mkdir(parents=True)
        
        with work_in(nested_tmp):
            assert os.getcwd() == str(nested_tmp.resolve())
            # Check that we can create a file here
            test_file = nested_tmp / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def mock_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    extension_class = simple_filter(mock_filter)

    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension (this triggers the registration logic in __init__)
    extension_instance = extension_class(env)

    # Assertions
    assert extension_class.__name__ == "mock_filter"
    assert "mock_filter" in env.filters
    assert env.filters["mock_filter"] == mock_filter
    
    # Verify the filter actually works within the environment
    template = env.from_string("{{ 'hello' | mock_filter }}")
    assert template.render() == "HELLO"

    # Verify it handles multiple registrations correctly
    def another_filter(x):
        return x + 1
    
    another_ext = simple_filter(another_filter)
    another_ext(env)
    assert "another_filter" in env.filters
    assert env.from_string("{{ 5 | another_filter }}").render() == 6
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_test_filter)
    
    # Create a Jinja environment
    env = Environment()
    
    # Initialize the extension with the environment
    extension = ExtensionClass(env)
    
    # Verify that the filter was added to the environment
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    
    # Verify functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    assert template.render() == "HELLO"
    
    # Verify that the class name was correctly updated
    assert ExtensionClass.__name__ == "my_test_filter"
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes directory and reverts back."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test with specific dirname
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
            
        # Assert we returned to original directory
        assert os.getcwd() == original_dir

    # Test with None (should not change anything)
    with work_in(None):
        assert os.getcwd() == original_dir

def test_work_in_error_handling():
    """Test that work_in reverts directory even if an error occurs inside the block."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        try:
            with work_in(tmp_path):
                assert os.getcwd() == str(tmp_path.resolve())
                raise ValueError("Simulated error")
        except ValueError:
            pass
            
        # Assert we still returned to original directory despite the exception
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #14
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_dir = os.getcwd()
    subdir = tmp_path / "test_sub"
    subdir.mkdir()

    # Test without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test with directory change
    with work_in(subdir):
        assert os.getcwd() == str(subdir.resolve())
    
    # Test return to original directory after context exit
    assert os.getcwd() == original_dir

    # Test with a non-existent directory (should raise FileNotFoundError)
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
    
    # Ensure we are still in the original directory after a failure
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a sample filter function
    def my_custom_filter(value):
        return value.upper()

    # Use the decorator/factory to create the Jinja2 extension class
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Initialize a Jinja2 environment
    env = Environment()
    
    # Instantiate the extension with the environment
    # This should trigger the registration of the filter in the environment
    extension_instance = ExtensionClass(env)
    
    # Assertions
    assert my_custom_filter.__name__ in env.filters
    assert env.filters[my_custom_filter.__name__] == my_custom_filter
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter actually works within the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory does not change when dirname is None
    original_cwd = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_cwd

    # Test 2: Verify working directory changes and reverts to original
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # After context manager exits, we must be back in the original directory
        assert os.getcwd() == original_cwd

    # Test 3: Verify working directory changes and reverts even if an error occurs inside
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        try:
            with work_in(tmp_path):
                assert os.getcwd() == str(tmp_path)
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #17
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test without arguments (should stay in original dir)
    with work_in():
        assert os.getcwd() == original_cwd
        
    # Test with directory argument
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
    
    # Ensure we returned to the original directory after context exit
    assert os.getcwd() == original_cwd

    # Test with a non-existent directory (should raise error but still restore CWD)
    bad_dir = tmp_path / "non_existent"
    try:
        with work_in(bad_dir):
            pass
    except FileNotFoundError:
        pass
    
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #18
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test behavior without arguments (should remain in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test behavior with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Should return to original directory after context exit
        assert os.getcwd() == original_dir

    # Test behavior with a nested directory structure
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        nested_path = root_path / "subdir"
        nested_path.mkdir()

        with work_in(nested_path):
            assert os.getcwd() == str(nested_path)
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #19
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory remains unchanged when no dirname is provided
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Verify working directory changes to the specified directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        # Verify we returned to original directory after context exit
        assert os.getcwd() == original_cwd

    # Test 3: Verify working directory changes even with string path input
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path_str = str(Path(tmp_dir).resolve())
        with work_in(tmp_path_str):
            assert os.getcwd() == tmp_path_str
        assert os.getcwd() == original_cwd

    # Test 4: Verify working directory reverts even if an error occurs inside the block
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        try:
            with work_in(tmp_path):
                assert os.getcwd() == str(tmp_path)
                raise ValueError("Simulated error")
        except ValueError:
            pass
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #20
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test original working directory is preserved
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Case 1: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # After exiting context, should be back to original
        assert os.getcwd() == original_cwd
        
        # Case 2: Using work_in with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test working directory remains same when no argument is provided
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test switching to a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the original directory after context exit
        assert os.getcwd() == original_dir

    # Test switching to a nested directory
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        nested_path = root_path / "subdir"
        nested_path.mkdir()
        
        original_dir = os.getcwd()
        with work_in(nested_path):
            assert os.getcwd() == str(nested_path)
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    FilterExtensionClass = simple_filter(my_test_filter)
    
    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the extension with the environment
    extension_instance = FilterExtensionClass(env)
    
    # Verify the extension is registered in the environment's filters
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    
    # Verify functionality via the environment
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    assert template.render() == "HELLO"
    
    # Verify the class name was correctly updated to the function name
    assert FilterExtensionClass.__name__ == "my_test_filter"
```


# LLM-generated content at query #23
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test with dirname provided
    with work_in(new_dir):
        assert Path(os.getcwd()).resolve() == new_dir.resolve()
    
    assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()

    # Test with None (should not change directory)
    with work_in(None):
        assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()
```


# LLM-generated content at query #24
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: No arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: With a specific directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the original directory after context exit
        assert os.getcwd() == original_dir

    # Test 3: Verify it handles nested changes correctly
    with tempfile.TemporaryDirectory() as tmp_dir:
        inner_dir = Path(tmp_dir) / "inner"
        inner_dir.mkdir()
        
        with work_in(tmp_dir):
            assert os.getcwd() == str(Path(tmp_dir).resolve())
            with work_in(inner_dir):
                assert os.getcwd() == str(inner_dir.resolve())
            assert os.getcwd() == str(Path(tmp_dir).resolve())
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #25
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Setup: Create a temporary directory and a subdirectory
    original_dir = os.getcwd()
    sub_dir = tmp_path / "sub_directory"
    sub_dir.mkdir()
    
    # Test 1: Context manager without arguments (should not change directory)
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Context manager with a directory (should change and revert)
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        
    # After exiting, we should be back in the original directory
    assert os.getcwd() == original_dir

    # Test 3: Verify nested usage
    with work_in(sub_dir):
        inner_dir = sub_dir / "inner"
        inner_dir.mkdir()
        with work_in(inner_dir):
            assert os.getcwd() == str(inner_dir.resolve())
        # Back to sub_dir
        assert os.getcwd() == str(sub_dir.resolve())
    
    # Back to original
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #26
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Context manager with a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            # Verify we can perform operations in the new directory
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # After exiting context, must return to original directory
        assert os.getcwd() == original_dir

    # Test 3: Verifying behavior with a nested path
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        sub_path = root_path / "subdir"
        sub_path.mkdir()
        
        with work_in(sub_path):
            assert os.getcwd() == str(sub_path)
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(s):
        return s.upper()

    # Use the decorator/factory to create the extension class
    ExtensionClass = simple_filter(my_custom_filter)

    # Create a Jinja environment
    env = Environment()

    # Instantiate the extension with the environment
    # The implementation of simple_filter attaches the filter to env.filters during __init__
    extension = ExtensionClass(env)

    # Assert that the filter is now present in the environment's filters
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter

    # Verify the functionality of the registered filter
    assert env.from_string("hello").render(my_custom_filter="hello") == "HELLO"
    
    # Verify the class name was updated correctly
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension = ExtensionClass(env)

    # Assertions
    assert my_custom_filter.__name__ in env.filters
    assert env.filters[my_custom_filter.__name__] == my_custom_filter
    assert ExtensionClass.__name__ == "my_custom_filter"
    
    # Verify the filter actually works within the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #29
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in changes directory and returns to original directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with a specific directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # Verify we returned to the original directory
        assert os.getcwd() == original_cwd

    # Test with None (should stay in current directory or handle no-op)
    with work_in(None):
        assert os.getcwd() == original_cwd

def test_work_in_error_handling():
    """Test that work_in restores CWD even if an error occurs inside the block."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        try:
            with work_in(tmp_path):
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
            
        # Verify we still returned to the original directory despite the exception
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #30
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify context manager restores original directory when no dirname is provided
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Verify context manager changes to the specified directory and restores on exit
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        assert os.getcwd() == original_dir

    # Test 3: Verify context manager works with string paths instead of Path objects
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_str_path = str(Path(tmp_dir).resolve())
        
        with work_in(tmp_str_path):
            assert os.getcwd() == tmp_str_path
        
        assert os.getcwd() == original_dir

    # Test 4: Verify error in block still triggers restoration of directory (contextlib behavior)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        try:
            with work_in(tmp_path):
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #31
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_dir = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test with dirname provided
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    # Verify return to original directory
    assert os.getcwd() == original_dir

    # Test without dirname (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test with a non-existent directory (should raise FileNotFoundError via os.chdir)
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
    
    # Verify even after failure, we still return to original dir
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Use the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Create a Jinja Environment
    env = Environment()
    
    # Instantiate the extension within the environment
    extension = ExtensionClass(env)
    
    # Verify that the filter was added to the environment's filters dictionary
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    # Verify functionality of the registered filter
    template = env.from_string("{{ 'hello'.my_custom_filter() }}")
    assert template.render() == "HELLO"

    # Verify that the class name matches the function name
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    extension_class = simple_filter(my_custom_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate and register the extension
    extension_instance = extension_class(env)

    # Verify the extension name was correctly assigned
    assert extension_class.__name__ == "my_custom_filter"

    # Verify the filter is actually registered in the environment
    assert "my_custom_filter" in env.filters
    
    # Verify the functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"

    # Verify it works with multiple calls/different values
    assert env.filters["my_custom_filter"]("test") == "TEST"
```


# LLM-generated content at query #34
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes and restores the working directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with no arguments (should stay in original dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test with directory argument (should change to tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test that it restores the original dir after context exit
        assert os.getcwd() == original_cwd

    # Final safety check for the environment state
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja Environment
    env = Environment()
    
    # Instantiate the generated extension class
    extension = ExtensionClass(env)

    # Assertions
    assert ExtensionClass.__name__ == "my_custom_filter"
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    # Verify the filter actually works within the context of the environment setup
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #36
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: Context manager without arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test case 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
        
        # Ensure we returned to the original directory after exiting context
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Test case 3: Verifying behavior with a subdirectory
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        sub_dir = root_path / "subdir"
        sub_dir.mkdir()
        
        original_dir_inside_test = os.getcwd()
        
        with work_in(root_path):
            assert Path(os.getcwd()).resolve() == root_path
            with work_in(sub_dir):
                assert Path(os.getcwd()).resolve() == sub_dir
            assert Path(os.getcwd()).resolve() == root_path
            
        assert Path(os.getcwd()).resolve() == Path(original_dir_inside_test).resolve()
```


# LLM-generated content at query #37
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    subdir = tmp_path / "test_dir"
    subdir.mkdir()

    # Test without dirname (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test with dirname (should change and revert)
    with work_in(subdir):
        assert os.getcwd() == str(subdir.resolve())
    
    assert os.getcwd() == original_cwd

    # Test with None explicitly passed
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #38
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: Context manager without dirname (should stay in current directory)
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test case 2: Context manager with dirname (should change and return)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        assert os.getcwd() == original_cwd

    # Test case 3: Context manager with a subdirectory
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()
        sub_path = root_path / "subdir"
        sub_path.mkdir()
        
        with work_in(root_path):
            assert os.getcwd() == str(root_path)
            with work_in(sub_path):
                assert os.getcwd() == str(sub_path)
            assert os.getcwd() == str(root_path)
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #39
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current dir)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
            
        # After exiting context, must return to original directory
        assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()

    # Test 3: Ensure it handles nested changes correctly (stack behavior)
    with tempfile.TemporaryDirectory() as dir1:
        path1 = Path(dir1).resolve()
        with tempfile.TemporaryDirectory() as dir2:
            path2 = Path(dir2).resolve()
            
            with work_in(path1):
                assert Path(os.getcwd()).resolve() == path1
                with work_in(path2):
                    assert Path(os.getcwd()).resolve() == path2
                assert Path(os.getcwd()).resolve() == path1
                
            assert Path(os.getcwd()).resolve() == Path(original_dir).resolve()
```


# LLM-generated content at query #40
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory does not change when no argument is provided
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir
    assert os.getcwd() == initial_dir

    # Test 2: Verify working directory changes and reverts to original
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            # Check if we can perform operations in the new directory
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Verify return to original directory after context manager exit
        assert os.getcwd() == initial_dir

    # Test 3: Verify behavior with a non-existent directory (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        with work_in("/non/existent/path/at/all"):
            pass
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Case 1: Using work_in without arguments (should stay in original dir)
        with work_in():
            assert os.getcwd() == original_dir
            
        # Case 2: Using work_in with a directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # After exiting the context manager, we should be back in the original dir
        assert os.getcwd() == original_dir

    # Final check to ensure no side effects leaked out
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_test_filter)

    # Initialize Jinja environment
    env = Environment()
    
    # Instantiate the extension
    extension = ExtensionClass(env)

    # Verify the filter was registered in the environment
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter

    # Verify functionality of the registered filter
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    assert template.render() == "HELLO"

    # Verify the extension class name matches the function name
    assert ExtensionClass.__name__ == "my_test_filter"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Use the decorator/factory to create the extension class
    ExtensionClass = simple_filter(my_test_filter)
    
    # Create a Jinja2 environment
    env = Environment()
    
    # Instantiate the extension with the environment
    # This is what the internal __init__ of SimpleFilterExtension does
    extension = ExtensionClass(env)

    # Assertions
    assert extension.__name__ == "my_test_filter"
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    
    # Verify the filter actually works within the environment
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    assert template.render() == "HELLO"

def test_simple_filter_multiple():
    def filter_a(x): return x
    def filter_b(x): return x

    env = Environment()
    
    # Test multiple registrations
    ext_a = simple_filter(filter_a)(env)
    ext_b = simple_filter(filter_b)(env)

    assert "filter_a" in env.filters
    assert "filter_b" in env.filters
```


# LLM-generated content at query #4
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_dir = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test context manager with a directory
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())

    # Test that it returns to the original directory
    assert os.getcwd() == original_dir

    # Test context manager without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test context manager with a path that doesn't exist (os.chdir will raise FileNotFoundError)
    non_existent = tmp_path / "ghost"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
    
    # Ensure we are still in the original directory after an error
    assert os.getcwd() == original_dir
```


# LLM-generated content at query #5
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a subdirectory to move into
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test with dirname provided
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        
    # Test that it returns to original directory
    assert os.getcwd() == original_cwd

    # Test without dirname (should stay in current dir)
    with work_in(None):
        assert os.getcwd() == original_cwd

    # Test with a nested directory
    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()
    with work_in(nested_dir):
        assert os.getcwd() == str(nested_dir.resolve())
    
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #6
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to switch into
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test 1: Context manager with dirname
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
    
    # Verify we returned to original directory
    assert os.getcwd() == original_cwd
    
    # Test 2: Context manager without dirname (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test 3: Context manager with a non-existent path (should raise error via os.chdir)
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent):
            pass
            
    # Verify we are still in original directory after exception
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #7
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Setup: create a temporary directory to move into
    subdir = tmp_path / "sub"
    subdir.mkdir()
    original_cwd = os.getcwd()

    # Test 1: Context manager without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test 2: Context manager with directory argument (should change and revert)
    with work_in(subdir):
        assert os.getcwd() == str(subdir.resolve())
        # Verify we are actually inside the dir by checking file existence
        test_file = subdir / "test.txt"
        test_file.write_text("hello")
        assert test_file.exists()

    # Test 3: Ensure we reverted to original directory after exiting block
    assert os.getcwd() == original_cwd

    # Test 4: Context manager with None (should behave like no arg)
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #8
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    """Test that work_in changes directory and restores it."""
    original_dir = os.getcwd()
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()

    # Test context manager without arguments (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test context manager with directory change
    with work_in(new_dir):
        assert os.getcwd() == str(new_dir.resolve())
    
    # Verify restoration after exit
    assert os.getcwd() == original_dir

def test_work_in_nested(tmp_path):
    """Test nested work_in calls."""
    original_dir = os.getcwd()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    with work_in(dir_a):
        assert os.getcwd() == str(dir_a.resolve())
        with work_in(dir_b):
            assert os.getcwd() == str(dir_b.resolve())
        assert os.getcwd() == str(dir_a.resolve())

    assert os.getcwd() == original_dir
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without dirname (should not change directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Context manager with dirname (should change and return directory)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            # Verify we can perform operations in the new directory
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # After exiting context, should be back to original directory
        assert os.getcwd() == original_dir

    # Test 3: Verifying it handles string paths as well as Path objects
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_str_path = str(Path(tmp_dir).resolve())
        with work_in(tmp_str_path):
            assert os.getcwd() == tmp_str_path
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Using work_in without arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test 2: Using work_in with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the original directory after context exit
        assert os.getcwd() == original_dir

    # Test 3: Verifying it works with nested context managers
    with tempfile.TemporaryDirectory() as tmp_dir1:
        tmp_path1 = Path(tmp_dir1).resolve()
        with tempfile.TemporaryDirectory() as tmp_dir2:
            tmp_path2 = Path(tmp_dir2).resolve()
            
            with work_in(tmp_path1):
                assert os.getcwd() == str(tmp_path1)
                with work_in(tmp_path2):
                    assert os.getcwd() == str(tmp_path2)
                assert os.getcwd() == str(tmp_path1)
            
            assert os.getcwd() == original_dir
```


# LLM-generated content at query #11
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        with work_in(tmp_path):
            assert Path(os.getcwd()).resolve() == tmp_path
        
        # Ensure we returned to the original directory after exiting context
        assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()

    # Test 3: Context manager with a nested subdirectory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(subdir):
            assert Path(os.getcwd()).resolve() == subdir
        
        assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()
```


# LLM-generated content at query #12
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test original directory preservation
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname).resolve()
        
        # Test context manager with a new directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
            
        # After exiting, should be back to original
        assert os.getcwd() == original_dir

    # Test context manager without arguments (should stay in current dir)
    with work_in():
        assert os.getcwd() == original_dir

    # Test with a non-existent directory (should raise FileNotFoundError via os.chdir)
    with pytest.raises(FileNotFoundError):
        with work_in("/non/existent/path/at/all"):
            pass
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Use the decorator/factory to create the Jinja extension class
    ExtensionClass = simple_filter(my_test_filter)
    
    # Initialize a Jinja environment
    env = Environment()
    
    # Instantiate the extension with the environment
    # This should trigger the registration of the filter
    extension = ExtensionClass(env)

    # Assertions
    assert ExtensionClass.__name__ == "my_test_filter"
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter
    assert env.filters["my_test_filter"]("hello") == "HELLO"
```


# LLM-generated content at query #14
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify working directory remains unchanged when no dirname is provided
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Verify working directory changes and reverts correctly
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure we returned to the original directory after context exit
        assert os.getcwd() == original_dir

    # Test 3: Verify working directory changes with a nested subdirectory
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        sub_dir = root_path / "subdir"
        sub_dir.mkdir()
        
        with work_in(root_path):
            assert os.getcwd() == str(root_path)
            with work_in(sub_dir):
                assert os.getcwd() == str(sub_dir)
            assert os.getcwd() == str(root_path)
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in correctly changes and restores the working directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Case 1: No argument provided (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Case 2: Argument provided (should change to tmp_path)
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # After context manager exits, should return to original
        assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Use the decorator/factory to create the extension class
    ExtensionClass = simple_filter(my_custom_filter)
    
    # Initialize a Jinja2 environment
    env = Environment()
    
    # Instantiate the extension with the environment
    extension_instance = ExtensionClass(env)
    
    # Verify the extension is registered in the environment's filters
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    # Verify functionality of the filter via the environment
    template = env.from_string("{{ 'hello'.my_custom_filter() }}")
    assert template.render() == "HELLO"
    
    # Verify the class name matches the function name as per implementation
    assert ExtensionClass.__name__ == "my_custom_filter"
```


# LLM-generated content at query #17
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Create a temporary directory to act as our target directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        original_cwd = os.getcwd()
        
        # 1. Test context manager without arguments (should not change directory)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # 2. Test context manager with a valid directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # 3. Test that it returns to the original directory after exiting the block
        assert os.getcwd() == original_cwd

        # 4. Test context manager with a sub-directory
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        with work_in(sub_dir):
            assert os.getcwd() == sub_dir
        
        assert os.getcwd() == original_cwd

    # Ensure we are back to the original directory at the end of the test
    os.chdir(original_cwd)
```


# LLM-generated content at query #18
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Create a temporary directory to act as our target workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir).resolve()
        original_cwd = os.getcwd()
        
        # 1. Test without argument (should stay in current dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # 2. Test with directory argument (should change to target)
        with work_in(target_path):
            assert os.getcwd() == str(target_path)
            
        # 3. Test that it restores the original directory after exiting context
        # even if we changed directories inside the block
        with work_in(target_path):
            os.chdir(tempfile.gettempdir())
            assert os.getcwd() != str(target_path)
        
        assert os.getcwd() == original_cwd

    # 4. Test with a non-existent directory (should raise FileNotFoundError via os.chdir)
    with pytest.raises(FileNotFoundError):
        with work_in("/non/existent/path/at/all"):
            pass
```


# LLM-generated content at query #19
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test case 1: Using work_in with a specific directory
    with work_in(sub_dir):
        assert Path(os.getcwd()).resolve() == sub_dir.resolve()
    
    # Verify we returned to the original directory after context exit
    assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()

    # Test case 2: Using work_in without arguments (should stay in current dir)
    with work_in():
        assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()

    # Test case 3: Verifying directory change even if an error occurs inside the block
    try:
        with work_in(sub_dir):
            raise ValueError("Simulated error")
    except ValueError:
        pass
    
    # Verify we still returned to original directory despite the exception
    assert Path(os.getcwd()).resolve() == Path(original_cwd).resolve()
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_custom_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_custom_filter)

    # Initialize Jinja2 environment
    env = Environment()
    
    # Instantiate the generated extension class
    extension_instance = ExtensionClass(env)

    # Assertions
    assert ExtensionClass.__name__ == "my_custom_filter"
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    # Verify the filter actually works within the context of the environment
    template = env.from_string("{{ 'hello' | my_custom_filter }}")
    assert template.render() == "HELLO"
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test case 1: No directory provided (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir

    # Test case 2: Directory provided (should change and return)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        # Ensure we returned to the original directory
        assert os.getcwd() == original_dir

    # Test case 3: Using a non-existent directory (should raise error via os.chdir)
    with pytest.raises(FileNotFoundError):
        with work_in("/non/existent/path/at/all/hopefully"):
            pass
```


# LLM-generated content at query #22
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Should return to initial directory after exiting context
        assert os.getcwd() == initial_dir

    # Test 3: Ensure it handles nested changes correctly
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        sub_path = root_path / "subdir"
        sub_path.mkdir()
        
        with work_in(root_path):
            assert os.getcwd() == str(root_path)
            with work_in(sub_path):
                assert os.getcwd() == str(sub_path)
            assert os.getcwd() == str(root_path)
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #23
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    original_cwd = os.getcwd()
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()
    
    # Test without arguments (should stay in same directory)
    with work_in():
        assert os.getcwd() == original_cwd

    # Test with directory argument (should change and revert)
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        
    assert os.getcwd() == original_cwd

    # Test with None (should behave like no arguments)
    with work_in(None):
        assert os.getcwd() == original_cwd
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def my_test_filter(value):
        return value.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(my_test_filter)

    # Initialize Jinja2 Environment
    env = Environment()

    # Instantiate the generated Extension class
    extension_instance = ExtensionClass(env)

    # Verify the extension was registered in the environment's filters
    assert "my_test_filter" in env.filters
    assert env.filters["my_test_filter"] == my_test_filter

    # Verify functionality through the registered filter
    template = env.from_string("{{ 'hello'.my_test_filter() }}")
    assert template.render() == "HELLO"

    # Verify class name was correctly assigned to the extension class
    assert ExtensionClass.__name__ == "my_test_filter"
```


# LLM-generated content at query #25
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Context manager without arguments (should stay in current directory)
    original_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test 2: Context manager with a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # Ensure it returns to the original directory after exiting context
        assert os.getcwd() == original_dir

    # Test 3: Verifying behavior with a nested directory structure
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        sub_path = root_path / "subdir"
        sub_path.mkdir()
        
        with work_in(sub_path):
            assert os.getcwd() == str(sub_path)
            # Ensure we can perform operations in the changed directory
            test_file = sub_path / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
        
        assert os.getcwd() == original_dir
```


# LLM-generated content at query #26
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()
    
    initial_dir = os.getcwd()
    
    # Test context manager without arguments (should stay in same dir)
    with work_in():
        assert os.getcwd() == initial_dir
        
    # Test context manager with directory argument
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        # Verify we can perform operations in the new dir
        temp_file = sub_dir / "test.txt"
        temp_file.write_text("hello")
        assert temp_file.exists()
        
    # Test that it returns to original directory after exit
    assert os.getcwd() == initial_dir

def test_work_in_with_none(tmp_path):
    initial_dir = os.getcwd()
    
    # Explicitly passing None should behave like the no-arg version
    with work_in(None):
        assert os.getcwd() == initial_dir
    
    assert os.getcwd() == initial_dir
```


# LLM-generated content at query #27
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Create a temporary directory to use as the target working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        original_cwd = os.getcwd()
        
        # Test 1: Context manager without arguments (should remain in original dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Context manager with a new directory
        # First, create a subdirectory inside tmpdir
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        
        with work_in(sub_dir):
            assert os.getcwd() == str(sub_dir)
            # Verify we can perform operations in the new directory
            test_file = sub_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        # Test 3: Ensure it returns to original directory after exiting context
        assert os.getcwd() == original_cwd

    # Cleanup is handled by TemporaryDirectory
```


# LLM-generated content at query #28
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in changes directory and restores it."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test context manager without arguments (should stay in original dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test context manager with new directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Verify we returned to the original directory after exiting block
        assert os.getcwd() == original_cwd

    # Final safety check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #29
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify current working directory remains unchanged when no dirname is provided
    original_cwd = os.getcwd()
    with work_in():
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Verify changing to a new directory and returning back
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        assert os.getcwd() != tmp_path
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
        
        # After context manager exits, should be back to original
        assert os.getcwd() == original_cwd

    # Test 3: Verify behavior with a non-existent directory (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        with work_in("/non/existent/path/that/does/not/exist"):
            pass
```


# LLM-generated content at query #30
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test original directory preservation and switching
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        
        # Case 1: Context manager without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_dir
            
        # Case 2: Context manager with target directory
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
            
            # Verify nested change (should return to tmp_path, not original)
            nested_dir = tmp_path / "subdir"
            nested_dir.mkdir()
            with work_in(nested_dir):
                assert os.getcwd() == str(nested_dir.resolve())
            
            assert os.getcwd() == str(tmp_path.resolve())
            
        # Final check: must return to original directory after context exit
        assert os.getcwd() == original_dir

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_str = str(Path(tmpdirname).resolve())
        with work_in(tmp_str):
            assert os.getcwd() == tmp_str
```


# LLM-generated content at query #31
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test working directory remains the same when no argument is provided
    initial_dir = os.getcwd()
    with work_in():
        assert os.getcwd() == initial_dir
    assert os.getcwd() == initial_dir

    # Test changing to a new directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        
        # Verify we are not in the tmp_path initially
        assert os.getcwd() != tmp_path
        
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Verify we returned to the initial directory after context exit
        assert os.getcwd() == initial_dir

    # Test changing to a nested directory
    with tempfile.TemporaryDirectory() as root_tmp:
        root_path = Path(root_tmp).resolve()
        nested_path = root_path / "subdir"
        nested_path.mkdir()
        
        with work_in(nested_path):
            assert os.getcwd() == nested_path
            
        assert os.getcwd() == initial_dir
```


# LLM-generated content at query #32
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Test 1: Verify initial working directory is preserved and no change happens when dirname is None
    original_cwd = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_cwd
    assert os.getcwd() == original_cwd

    # Test 2: Verify changing to a new directory and returning back
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        assert os.getcwd() == str(original_path := Path(original_cwd).resolve())

    # Test 3: Verify functionality with a nested directory structure
    with tempfile.TemporaryDirectory() as root_tmp:
        nested_dir = Path(root_tmp) / "subdir" / "deep"
        nested_dir.mkdir(parents=True)
        
        with work_in(nested_dir):
            assert os.getcwd() == str(nested_dir.resolve())
            # Verify we can perform operations in the new directory
            test_file = nested_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
            
        assert os.getcwd() == str(Path(original_cwd).resolve())

    # Test 4: Verify error handling (if directory does not exist, it should raise FileNotFoundError)
    non_existent_dir = Path(tempfile.mkdtemp()) / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from jinja2 import Environment

def test_simple_filter():
    # Define a dummy filter function
    def mock_uppercase(s):
        return s.upper()

    # Apply the decorator/wrapper
    ExtensionClass = simple_filter(mock_uppercase)
    
    # Create a Jinja environment
    env = Environment()
    
    # Instantiate the extension class with the environment
    extension_instance = ExtensionClass(env)
    
    # Verify the filter was actually added to the environment's filters dictionary
    assert "mock_uppercase" in env.filters
    assert env.filters["mock_uppercase"] == mock_uppercase
    
    # Verify functionality through the environment
    template = env.from_string("{{ 'hello'.mock_uppercase() }}")
    # Note: simple_filter as implemented adds it to filters, 
    # but usage in template depends on how Jinja handles filter names vs method calls.
    # In standard Jinja, filters are applied via | symbol.
    template_pipe = env.from_string("{{ 'hello' | mock_uppercase }}")
    assert template_pipe.render() == "HELLO"

    # Verify the class name was updated correctly
    assert ExtensionClass.__name__ == "mock_uppercase"
```


# LLM-generated content at query #34
#--------------------------

```python
import os
from pathlib import Path

def test_work_in(tmp_path):
    # Create a temporary directory to move into
    sub_dir = tmp_path / "test_subdir"
    sub_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    # Test context manager without argument (should stay in same dir)
    with work_in():
        assert os.getcwd() == original_cwd
        
    # Test context manager with directory argument
    with work_in(sub_dir):
        assert os.getcwd() == str(sub_dir.resolve())
        
    # Verify we returned to the original directory after exit
    assert os.getcwd() == original_cwd

    # Test context manager with path as string
    with work_in(str(tmp_path)):
        assert os.getcwd() == str(tmp_path.resolve())
        
    # Final check
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_simple_filter():
    # Define a dummy filter function
    def my_dummy_filter(value):
        return value.upper()

    # Create a mock Jinja2 Environment
    mock_env = MagicMock()
    mock_env.filters = {}

    # Apply the decorator/wrapper
    extension_class = simple_filter(my_dummy_filter)

    # Instantiate the generated extension class with our mock environment
    extension_instance = extension_class(mock_env)

    # Assertions
    assert extension_class.__name__ == "my_dummy_filter"
    assert "my_dummy_filter" in mock_env.filters
    assert mock_env.filters["my_dummy_filter"] == my_dummy_filter
```


# LLM-generated content at query #36
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    """Test that work_in changes the directory and returns to the original one."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test with no argument (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test changing to a new directory
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Test that it reverts after the context manager exits
        assert os.getcwd() == original_cwd

    # Final check to ensure no side effects leaked
    assert os.getcwd() == original_cwd
```


# LLM-generated content at query #37
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest

def test_work_in():
    # Get current working directory to verify return trip
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        
        # Test 1: Context manager without arguments (should stay in same dir)
        with work_in():
            assert os.getcwd() == original_cwd
            
        # Test 2: Context manager with directory change
        with work_in(tmp_path):
            assert os.getcwd() == tmp_path
            
        # Verify that after exiting the context, we are back to original dir
        assert os.getcwd() == original_cwd

    # Final check to ensure no side effects leaked out of the test
    assert os.getcwd() == original_cwd
```


