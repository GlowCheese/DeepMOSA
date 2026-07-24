####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter():
    # Define a simple test filter function
    def test_filter(value):
        return value.upper() + "_FILTERED"
    
    # Apply the decorator
    FilterExtension = simple_filter(test_filter)
    
    # Create a test environment
    env = StrictEnvironment()
    
    # Initialize the extension with the environment
    extension = FilterExtension(env)
    
    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter
    
    # Test that the filter works correctly through the environment
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO_FILTERED"
    
    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == test_filter.__name__
    
    # Test with a different filter function
    def another_filter(value):
        return value.lower() + "_processed"
    
    AnotherExtension = simple_filter(another_filter)
    assert AnotherExtension.__name__ == another_filter.__name__


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir_path):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'test' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_test"


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even if exception occurs
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    # Test 1: Context manager changes directory and returns to original
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory after context manager exits
        assert os.getcwd() == original_dir
    
    # Test 2: Context manager with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test 3: Nested context managers work correctly
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                
                # Should return to tmpdir1 after inner context exits
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            
            # Should return to original after outer context exits
            assert os.getcwd() == original_dir
    
    # Test 4: Exception within context manager still returns to original directory
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still return to original directory even with exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #11
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to create the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory specified
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_tmp = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == current_in_tmp
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir


# LLM-generated content at query #14
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert "test_filter" in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == "test_filter"

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert "another_filter" in env2.filters

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in("subdir"):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #18
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter is registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "filtered_hello"

    # Verify the extension class name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        # Should return to original directory
        assert os.getcwd() == original_dir

    # Test with None (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir

    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Should have returned to original directory even after exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir


# LLM-generated content at query #21
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Test the extension's name attribute
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #22
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    # Test 1: Context manager changes directory and returns to original
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
    
    # Test 2: With None parameter (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test 3: Nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                
                # Should return to tmpdir1
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            
            # Should return to original directory
            assert os.getcwd() == original_dir
    
    # Test 4: Exception handling - should still return to original directory
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should have returned to original directory even with exception
        assert os.getcwd() == original_dir
    
    # Test 5: With Path object as input
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #24
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #27
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to create the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())

        # Should return to original directory
        assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())

        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir

    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #29
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test function to be used as a filter
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #30
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #31
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter decorator creates a proper Jinja2 extension
    from jinja2 import Environment
    
    # Define a simple filter function
    def test_filter(value):
        return f"filtered_{value}"
    
    # Apply the decorator
    FilterExtension = simple_filter(test_filter)
    
    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])
    
    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter
    
    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"
    
    # Test that the extension has the correct name
    assert FilterExtension.__name__ == 'test_filter'
    
    # Test with a different filter function
    def another_filter(value):
        return value.upper()
    
    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert 'another_filter' in env2.filters
    assert env2.filters['another_filter'] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #33
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #34
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'data' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_data"


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #36
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            
            # Should return to tmpdir
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2_path = Path(tmpdir2)
            
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(tmpdir2_path)
                
                assert os.getcwd() == str(tmpdir_path)
            
            assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #39
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Test extension name
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #40
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "filtered_hello"

    # Test the extension name
    assert FilterExtension.__name__ == "test_filter"

    # Test with a different filter
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter
    
    template2 = env2.from_string("{{ 'test' | another_filter }}")
    assert template2.render() == "TEST"


# LLM-generated content at query #41
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test function to be used as a filter
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #42
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify extension name was set correctly
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])

    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #43
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())

        # Should return to original directory
        assert os.getcwd() == original_dir

    # Test with None (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())

        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                
                # Should return to tmpdir1
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            
            # Should return to original directory
            assert os.getcwd() == original_dir

    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should have returned to original directory even with exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir

    # Test that directory is restored even if exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        assert os.getcwd() == original_dir

    # Test with string path
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #46
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            # Should return to tmpdir after inner context
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory after all contexts
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #47
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #48
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            # Should return to tmpdir
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #49
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'data' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_data"
    assert "custom_filter" in env2.filters
    assert env2.filters["custom_filter"] is custom_filter


# LLM-generated content at query #50
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    extension_instance = test_filter(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "TEST"


# LLM-generated content at query #51
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Verify the extension class was created with correct name
    assert test_filter.__name__ == "SimpleFilterExtension"
    assert test_filter.__name__ == "test_filter"

    # Create a test environment and add the extension
    env = StrictEnvironment(extensions=[test_filter])
    
    # Verify the filter was registered in the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter.__wrapped__

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Test with different input values
    template2 = env.from_string("{{ number | test_filter }}")
    result2 = template2.render(number=42)
    assert result2 == "filtered_42"

    # Verify the extension can be instantiated
    extension_instance = test_filter(env)
    assert isinstance(extension_instance, test_filter)


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #53
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    from jinja2 import Environment
    
    # Define a simple filter function
    def test_filter(value):
        return value.upper()
    
    # Apply the decorator
    SimpleFilterExtension = simple_filter(test_filter)
    
    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[SimpleFilterExtension])
    
    # Test that the filter was registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ "hello" | test_filter }}')
    result = template.render()
    assert result == 'HELLO'
    
    # Test the extension name
    assert SimpleFilterExtension.__name__ == 'test_filter'
    
    # Test with another filter
    def another_filter(value):
        return value.lower()
    
    AnotherFilterExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherFilterExtension])
    
    assert 'another_filter' in env2.filters
    assert env2.filters['another_filter'] is another_filter
    
    template2 = env2.from_string('{{ "WORLD" | another_filter }}')
    result2 = template2.render()
    assert result2 == 'world'


# LLM-generated content at query #54
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Verify final return to original
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #55
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #56
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify extension name was set correctly
    assert FilterExtension.__name__ == "test_filter"

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #57
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #58
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            
            # Should return to tmpdir
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test with a string directory
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test that it returns to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir_path):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'test' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_test"


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        with work_in(None):
            assert os.getcwd() == original_dir
        
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(subdir):
            assert os.getcwd() == str(subdir)
        
        assert os.getcwd() == original_dir
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())

        # Should return to original directory
        assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())

        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                
                # Should return to tmpdir1
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            
            # Should return to original
            assert os.getcwd() == original_dir

    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should have returned to original even with exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir

    # Test that it returns to original directory even if exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify final return to original directory
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir_path):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory specified
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #11
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "filtered_hello"

    # Verify the extension class name
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter
    assert AnotherExtension.__name__ == "another_filter"


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #15
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()
    
    # Create a test environment
    from jinja2 import Environment
    
    env = Environment(extensions=[test_filter])
    
    # Verify the filter was registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] == test_filter
    
    # Test the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == 'HELLO'
    
    # Verify the extension class name matches the filter function name
    assert test_filter.__name__ == 'SimpleFilterExtension'
    
    # Test with another filter
    @simple_filter
    def double(value):
        return value * 2
    
    env2 = Environment(extensions=[double])
    assert 'double' in env2.filters
    assert env2.filters['double'] == double
    
    template2 = env2.from_string("{{ 5 | double }}")
    result2 = template2.render()
    assert result2 == '10'


# LLM-generated content at query #16
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'data' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_data"


# LLM-generated content at query #17
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function
    def test_filter(value):
        return value.upper()

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test that the filter works correctly
    template = env.from_string('{{ "hello" | test_filter }}')
    result = template.render()
    assert result == 'HELLO'

    # Verify the extension name is set correctly
    assert FilterExtension.__name__ == 'test_filter'

    # Test with a different filter
    def another_filter(value):
        return value * 2

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert 'another_filter' in env2.filters
    assert env2.filters['another_filter'] is another_filter
    
    template2 = env2.from_string('{{ "ab" | another_filter }}')
    result2 = template2.render()
    assert result2 == 'abab'


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #19
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test filter functionality
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify extension name
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #20
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    @simple_filter
    def custom_filter(value):
        return f"processed_{value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'test' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed_test"


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify still in original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory specified
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            # Should return to tmpdir after inner context
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original after all contexts
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir_path):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in("subdir"):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Verify we're still in original directory
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmp_path.resolve())
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            
            assert os.getcwd() == str(tmp_path.resolve())
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - directory should still be restored
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmp_path.resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(str(subdir)):
            assert os.getcwd() == str(subdir)
        
        # Should return to original directory even after exception
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        another_dir = tmpdir_path / "another"
        another_dir.mkdir()
        
        with work_in(another_dir):
            assert os.getcwd() == str(another_dir)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Verify final return to original
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that it returns to original directory even if exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir


# LLM-generated content at query #33
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    assert test_filter.__name__ == "SimpleFilterExtension"
    assert hasattr(test_filter, "__init__")

    env2 = Environment()
    extension_instance = test_filter(env2)
    assert "test_filter" in env2.filters
    assert env2.filters["test_filter"]("test") == "TEST"


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #36
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function to test
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "filtered_hello"

    # Verify the extension class has the expected name
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter to ensure isolation
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert "another_filter" in env2.filters
    assert "test_filter" not in env2.filters
    assert env2.filters["another_filter"] is another_filter

    template2 = env2.from_string("{{ 'world' | another_filter }}")
    result2 = template2.render()
    assert result2 == "WORLD"


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir
    
    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == str(Path(tmpdir1).resolve())
                with work_in(tmpdir2):
                    assert os.getcwd() == str(Path(tmpdir2).resolve())
                assert os.getcwd() == str(Path(tmpdir1).resolve())
            assert os.getcwd() == original_dir
    
    # Test that directory is restored even if exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(Path(tmpdir).resolve())
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        assert os.getcwd() == original_dir


# LLM-generated content at query #38
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    @simple_filter
    def test_filter(value):
        return value.upper()

    env = Environment(extensions=[test_filter])
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    @simple_filter
    def custom_filter(value):
        return f"processed: {value}"

    env2 = Environment(extensions=[custom_filter])
    template2 = env2.from_string("{{ 'test' | custom_filter }}")
    result2 = template2.render()
    assert result2 == "processed: test"
    assert "custom_filter" in env2.filters
    assert env2.filters["custom_filter"] is custom_filter


# LLM-generated content at query #39
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()
    
    # Create a test environment
    env = StrictEnvironment()
    
    # Create and initialize the extension
    extension_class = test_filter
    extension = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter
    
    # Test that the filter works correctly
    result = env.filters['test_filter']('hello')
    assert result == 'HELLO'
    
    # Verify extension class name matches filter function name
    assert extension_class.__name__ == 'test_filter'


# LLM-generated content at query #40
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #41
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a valid directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == os.path.realpath(tmpdir)
        # Should return to original directory
        assert os.getcwd() == original_dir

    # Test with None (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == os.path.realpath(str(tmp_path))
        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == os.path.realpath(tmpdir1)
                with work_in(tmpdir2):
                    assert os.getcwd() == os.path.realpath(tmpdir2)
                assert os.getcwd() == os.path.realpath(tmpdir1)
            assert os.getcwd() == original_dir

    # Test exception handling - should still return to original directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        assert os.getcwd() == original_dir


# LLM-generated content at query #42
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in(subdir):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        # Test with Path object
        with work_in(Path(tmpdir)):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #43
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()

    # Test with a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == os.path.realpath(tmpdir)
        assert os.getcwd() == original_dir

    # Test with None (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == os.path.realpath(tmpdir)
        assert os.getcwd() == original_dir

    # Test nested context managers
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == os.path.realpath(tmpdir1)
                with work_in(tmpdir2):
                    assert os.getcwd() == os.path.realpath(tmpdir2)
                assert os.getcwd() == os.path.realpath(tmpdir1)
            assert os.getcwd() == original_dir

    # Test exception handling - directory should still be restored
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == os.path.realpath(tmpdir)
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert os.getcwd() == original_dir


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should not change directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Verify we're still in original directory
        assert os.getcwd() == original_dir
        
        # Test with Path object
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmp_path2 = Path(tmpdir2)
            
            with work_in(tmpdir):
                assert os.getcwd() == str(tmp_path.resolve())
                
                with work_in(tmpdir2):
                    assert os.getcwd() == str(tmp_path2.resolve())
                
                # Should return to first temp directory
                assert os.getcwd() == str(tmp_path.resolve())
            
            # Should return to original directory
            assert os.getcwd() == original_dir


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile

    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory path
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #46
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #47
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory specified
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test with string path
        with work_in(str(tmpdir)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #48
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Test the extension class name was set correctly
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with a directory
        with work_in(tmpdir):
            assert os.getcwd() == str(Path(tmpdir).resolve())
        
        # Should return to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test with Path object
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj.resolve())
        
        # Verify still in original directory
        assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            current_in_first = os.getcwd()
            with work_in("subdir"):
                assert os.getcwd() == str(subdir.resolve())
            assert os.getcwd() == current_in_first
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #50
#--------------------------

```python
def test_work_in():
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test with directory specified
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
        
        # Verify we returned to original directory
        assert os.getcwd() == original_dir
        
        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir
        
        # Test nested context managers
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        
        with work_in(tmpdir):
            assert os.getcwd() == str(tmpdir_path)
            
            with work_in(subdir):
                assert os.getcwd() == str(subdir)
            
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir
        
        # Test exception handling - should still return to original directory
        try:
            with work_in(tmpdir):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #51
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test function to be used as a filter
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter functionality
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify extension name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #52
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test function to be used as a filter
    def test_filter(value):
        return value.upper()

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == 'HELLO'

    # Verify extension name is set correctly
    assert FilterExtension.__name__ == 'test_filter'

    # Test with a different filter function
    def another_filter(value):
        return value * 2

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert 'another_filter' in env2.filters
    assert env2.filters['another_filter'] is another_filter
    
    template2 = env2.from_string("{{ 'ab' | another_filter }}")
    result2 = template2.render()
    assert result2 == 'abab'


# LLM-generated content at query #53
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function to test
    def test_filter(value):
        return value.upper()

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "HELLO"

    # Verify the extension class has the expected name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value * 2

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'ab' | another_filter }}")
    result2 = template2.render()
    assert result2 == "abab"


# LLM-generated content at query #54
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == "test_filter"

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])

    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter

    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


# LLM-generated content at query #55
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple filter function to test
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Test that the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    result = template.render()
    assert result == "filtered_hello"

    # Test the extension name
    assert FilterExtension.__name__ == "test_filter"

    # Test with another filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    assert "another_filter" in env2.filters
    assert env2.filters["another_filter"] is another_filter

    template2 = env2.from_string("{{ 'test' | another_filter }}")
    assert template2.render() == "TEST"


# LLM-generated content at query #56
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Test function to be used as a filter
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify extension name is set correctly
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #57
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to get the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    assert template2.render() == "HELLO"


# LLM-generated content at query #58
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment

    # Define a simple test filter function
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator to create the extension class
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment with the extension
    env = Environment(extensions=[FilterExtension])

    # Verify the filter was registered with the correct name
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly in template rendering
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == test_filter.__name__

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherExtension = simple_filter(another_filter)
    env2 = Environment(extensions=[AnotherExtension])
    
    assert another_filter.__name__ in env2.filters
    assert env2.filters[another_filter.__name__] is another_filter
    
    template2 = env2.from_string("{{ 'hello' | another_filter }}")
    result2 = template2.render()
    assert result2 == "HELLO"


