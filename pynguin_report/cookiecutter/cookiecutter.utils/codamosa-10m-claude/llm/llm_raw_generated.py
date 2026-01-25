####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in contexts."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                
                assert os.getcwd() == tmpdir1
            
            assert os.getcwd() == original_dir


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory after exiting
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that directory is still the original after exiting
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in context manager restores directory even on exception."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in restores directory even when exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    
    # Test that directory is restored after exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in works with Path objects
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in context manager with string path."""
    original_dir = os.getcwd()
    test_dir = str(tmp_path / "test_workdir")
    Path(test_dir).mkdir()
    
    # Test that work_in works with string paths
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Apply the decorator
    extension_class = simple_filter(uppercase)
    
    # Verify it's a class
    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)
    
    # Verify the extension name matches the function name
    assert extension_class.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def lowercase(value):
        """Convert string to lowercase."""
        return value.lower()
    
    def reverse(value):
        """Reverse a string."""
        return value[::-1]
    
    # Apply decorators
    lowercase_ext = simple_filter(lowercase)
    reverse_ext = simple_filter(reverse)
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test both filters work
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Test that directory is restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test changing to a different directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Verify we're back to the original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify we're back to the original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager
        with work_in(temp_dir):
            # Inside context, should be in temp_dir
            assert os.getcwd() == temp_dir
        
        # After exiting context, should be back in original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_none():
    """Test work_in with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        # Should remain in original directory
        assert os.getcwd() == original_dir
    
    # Should still be in original directory
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Should be back in original directory despite exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify directory is restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test work_in with nested context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            with work_in(temp_dir1):
                assert os.getcwd() == temp_dir1
                with work_in(temp_dir2):
                    assert os.getcwd() == temp_dir2
                assert os.getcwd() == temp_dir1
            assert os.getcwd() == original_dir


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory after exiting
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in context manager with None dirname."""
    original_dir = os.getcwd()
    
    # Test that work_in with None stays in current directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that original directory is restored
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in restores directory even if exception is raised
    with contextlib.suppress(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    
    # Test that original directory is restored after exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in works with Path objects
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in context manager with string path."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in works with string paths
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #9
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a Jinja2 extension with filter."""
    
    def uppercase_filter(text):
        """Simple test filter that converts text to uppercase."""
        return text.upper()
    
    # Create the extension class
    FilterExtension = simple_filter(uppercase_filter)
    
    # Verify it's a class and inherits from Extension
    assert isinstance(FilterExtension, type)
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'uppercase_filter'
    
    # Create a Jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|uppercase_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators create independent extensions."""
    
    def lowercase_filter(text):
        return text.lower()
    
    def reverse_filter(text):
        return text[::-1]
    
    LowercaseExtension = simple_filter(lowercase_filter)
    ReverseExtension = simple_filter(reverse_filter)
    
    env = StrictEnvironment()
    LowercaseExtension(env)
    ReverseExtension(env)
    
    assert 'lowercase_filter' in env.filters
    assert 'reverse_filter' in env.filters
    
    template = env.from_string('{{ text|lowercase_filter|reverse_filter }}')
    result = template.render(text='HELLO')
    assert result == 'olleh'


def test_simple_filter_with_arguments():
    """Test simple_filter with a filter function that takes arguments."""
    
    def repeat_filter(text, times=2):
        return text * times
    
    RepeatExtension = simple_filter(repeat_filter)
    
    env = StrictEnvironment()
    RepeatExtension(env)
    
    assert 'repeat_filter' in env.filters
    
    template = env.from_string('{{ text|repeat_filter(3) }}')
    result = template.render(text='ab')
    assert result == 'ababab'


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #11
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase_filter(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Decorate the function with simple_filter
    FilterExtension = simple_filter(uppercase_filter)
    
    # Verify the extension class is created
    assert FilterExtension is not None
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the filter function name
    assert FilterExtension.__name__ == 'uppercase_filter'
    
    # Create a jinja2 environment and register the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter is registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|uppercase_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_filter(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_filter)
    
    assert FilterExtension.__name__ == 'reverse_filter'
    
    env = StrictEnvironment(extensions=[FilterExtension])
    assert 'reverse_filter' in env.filters
    
    template = env.from_string('{{ text|reverse_filter }}')
    result = template.render(text='world')
    assert result == 'dlrow'


def test_simple_filter_with_multiple_arguments():
    """Test simple_filter with a filter that takes multiple arguments."""
    def repeat_filter(value, times=2):
        """Repeat a string multiple times."""
        return str(value) * times
    
    FilterExtension = simple_filter(repeat_filter)
    
    env = StrictEnvironment(extensions=[FilterExtension])
    assert 'repeat_filter' in env.filters
    
    template = env.from_string('{{ text|repeat_filter(3) }}')
    result = template.render(text='x')
    assert result == 'xxx'


# LLM-generated content at query #12
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter correctly wraps a function in a Jinja2 extension."""
    # Define a simple filter function
    def uppercase_filter(value):
        """Convert a string to uppercase."""
        return str(value).upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase_filter)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a Jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_filter(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    extension_class = simple_filter(reverse_filter)
    
    assert extension_class.__name__ == 'reverse_filter'
    assert issubclass(extension_class, Extension)
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_filter' in env.filters
    result = env.filters['reverse_filter']('abc')
    assert result == 'cba'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    ext1 = simple_filter(add_prefix)
    ext2 = simple_filter(add_suffix)
    
    env = StrictEnvironment()
    ext1(env)
    ext2(env)
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #13
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    def my_custom_filter(value):
        """Test filter that uppercases a string."""
        return value.upper()
    
    # Create the extension class
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify the extension class name matches the function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Create a jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|my_custom_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Test filter that reverses a string."""
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    template = env.from_string('{{ text|reverse_string }}')
    result = template.render(text='hello')
    assert result == 'olleh'


def test_simple_filter_with_multiple_arguments():
    """Test simple_filter with a filter that takes multiple arguments."""
    def repeat_string(value, times=2):
        """Test filter that repeats a string."""
        return value * times
    
    FilterExtension = simple_filter(repeat_string)
    
    env = StrictEnvironment()
    FilterExtension(env)
    
    template = env.from_string('{{ text|repeat_string(3) }}')
    result = template.render(text='x')
    assert result == 'xxx'


# LLM-generated content at query #14
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter correctly wraps a function in a Jinja2 extension."""
    # Define a simple filter function
    def uppercase_filter(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase_filter)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and initialize the extension
    env = StrictEnvironment()
    extension = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_filter(value):
        """Reverse the value."""
        return str(value)[::-1]
    
    extension_class = simple_filter(reverse_filter)
    
    assert extension_class.__name__ == 'reverse_filter'
    
    env = StrictEnvironment()
    extension = extension_class(env)
    
    assert 'reverse_filter' in env.filters
    result = env.filters['reverse_filter']('abc')
    assert result == 'cba'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    prefix_extension = simple_filter(add_prefix)
    suffix_extension = simple_filter(add_suffix)
    
    env = StrictEnvironment()
    prefix_ext = prefix_extension(env)
    suffix_ext = suffix_extension(env)
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    import tempfile
    
    # Get the original working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory to work in
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test 1: work_in changes to the specified directory
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        # Test 2: work_in restores the original directory after exiting
        assert os.getcwd() == original_cwd
        
        # Test 3: work_in with None argument should not change directory
        with work_in(None):
            assert os.getcwd() == original_cwd
        
        # Test 4: work_in restores directory even if an exception occurs
        try:
            with work_in(tmpdir_path):
                assert os.getcwd() == str(tmpdir_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_cwd
        
        # Test 5: work_in works with string path
        with work_in(str(tmpdir_path)):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_cwd


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even on exception."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Test that directory is restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that we change to the new directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that we return to the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None dirname."""
    original_dir = os.getcwd()
    
    with work_in(None):
        # Should remain in the same directory
        assert os.getcwd() == original_dir
    
    # Should still be in original directory
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even on exception."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #18
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a Jinja2 extension with the filter registered."""
    # Define a simple filter function
    def uppercase(value):
        return value.upper()
    
    # Create the extension using simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it's a class that extends Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and initialize the extension with it
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Verify the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        return value[::-1]
    
    # Create the extension
    FilterExtension = simple_filter(reverse_string)
    
    # Verify extension name
    assert FilterExtension.__name__ == 'reverse_string'
    
    # Create environment and extension
    env = StrictEnvironment()
    FilterExtension(env)
    
    # Verify filter works
    result = env.filters['reverse_string']('hello')
    assert result == 'olleh'


def test_simple_filter_multiple_extensions():
    """Test that multiple filters can be created and used together."""
    def add_prefix(value):
        return f'prefix_{value}'
    
    def add_suffix(value):
        return f'{value}_suffix'
    
    # Create both extensions
    PrefixExtension = simple_filter(add_prefix)
    SuffixExtension = simple_filter(add_suffix)
    
    # Create environment
    env = StrictEnvironment()
    PrefixExtension(env)
    SuffixExtension(env)
    
    # Verify both filters are registered
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    
    # Verify both filters work
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in restores directory even on exception
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("test exception")
        
        # Verify directory was restored despite exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_cwd


def test_work_in_with_none():
    """Test work_in with None argument stays in current directory."""
    original_cwd = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_cwd
    
    assert os.getcwd() == original_cwd


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory even after exception
        assert os.getcwd() == original_cwd


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_cwd


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                assert os.getcwd() == tmpdir1
            assert os.getcwd() == original_cwd


# LLM-generated content at query #21
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator wraps a function in a Jinja2 extension."""
    # Define a simple test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return str(value).upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(uppercase)
    
    # Verify the extension class was created with correct name
    assert FilterExtension.__name__ == 'uppercase'
    assert issubclass(FilterExtension, Extension)
    
    # Create a Jinja2 environment with the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] == uppercase
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|uppercase }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test that multiple simple_filter extensions can be used together."""
    def reverse_string(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    def add_prefix(value):
        """Add a prefix to a string."""
        return f'PREFIX_{value}'
    
    ReverseExtension = simple_filter(reverse_string)
    PrefixExtension = simple_filter(add_prefix)
    
    env = StrictEnvironment(extensions=[ReverseExtension, PrefixExtension])
    
    assert 'reverse_string' in env.filters
    assert 'add_prefix' in env.filters
    
    template = env.from_string('{{ text|add_prefix|reverse_string }}')
    result = template.render(text='test')
    assert result == 'tset_XIFWERP'


def test_simple_filter_preserves_function_behavior():
    """Test that simple_filter preserves the original function behavior."""
    def multiply_by_two(value):
        """Multiply value by two."""
        return int(value) * 2
    
    FilterExtension = simple_filter(multiply_by_two)
    env = StrictEnvironment(extensions=[FilterExtension])
    
    template = env.from_string('{{ num|multiply_by_two }}')
    result = template.render(num=5)
    assert result == '10'


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    import tempfile
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if exception occurs inside context
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be back in original directory
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    import os
    from pathlib import Path
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager to change to temp directory
        with work_in(temp_path):
            # Inside context, we should be in temp_dir
            assert os.getcwd() == str(temp_path)
        
        # After exiting context, we should be back in original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test the work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # When dirname is None, should stay in current directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Should still be in original directory after exiting
    assert os.getcwd() == original_dir


def test_work_in_returns_to_original_on_exception():
    """Test that work_in returns to original directory even on exception."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Verify exception doesn't prevent return to original directory
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be back in original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test the work_in context manager with string path."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use string path instead of Path object
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores original directory even when exception is raised."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should restore original directory despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in context manager works with string paths."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if an exception occurs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory despite exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even on exception."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                assert os.getcwd() == tmpdir1
            assert os.getcwd() == original_dir


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


def test_work_in_with_string_path():
    """Test work_in context manager works with string paths."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_none():
    """Test work_in context manager with None dirname stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        
        # Verify directory is restored even after exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_path_object():
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension with a filter."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    
    # Verify the filter function works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'
    assert env.filters['uppercase']('test') == 'TEST'


def test_simple_filter_with_jinja_template():
    """Test that simple_filter works within a jinja2 template."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = StrictEnvironment(extensions=[extension_class])
    
    # Verify the filter is available
    assert 'reverse_string' in env.filters
    
    # Test with a template
    template = env.from_string("{{ 'hello' | reverse_string }}")
    result = template.render()
    assert result == 'olleh'


def test_simple_filter_multiple_filters():
    """Test creating multiple simple_filter extensions."""
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    ext1 = simple_filter(add_prefix)
    ext2 = simple_filter(add_suffix)
    
    env = StrictEnvironment(extensions=[ext1, ext2])
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None dirname."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that directory is still the same after exiting
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Test that directory is restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    import os
    from pathlib import Path
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that directory is restored even if exception occurs
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite the exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    import os
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with string path
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    import tempfile
    
    # Get the original working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory to change into
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_cwd
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_cwd
    
    assert os.getcwd() == original_cwd


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    import os
    from pathlib import Path
    import tempfile
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_cwd


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    import os
    from pathlib import Path
    import tempfile
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_cwd


# LLM-generated content at query #33
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    # Define a test filter function
    def test_upper(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Decorate the function with simple_filter
    FilterExtension = simple_filter(test_upper)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'test_upper'
    
    # Create a StrictEnvironment and initialize the extension with it
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter function was registered in the environment
    assert 'test_upper' in env.filters
    assert env.filters['test_upper'] is test_upper
    
    # Test that the filter actually works
    result = env.filters['test_upper']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test simple_filter works with multiple different filter functions."""
    def test_lower(value):
        """Convert value to lowercase."""
        return value.lower()
    
    def test_reverse(value):
        """Reverse a string."""
        return value[::-1]
    
    # Create extensions for both filters
    LowerExtension = simple_filter(test_lower)
    ReverseExtension = simple_filter(test_reverse)
    
    # Verify correct names
    assert LowerExtension.__name__ == 'test_lower'
    assert ReverseExtension.__name__ == 'test_reverse'
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    LowerExtension(env)
    ReverseExtension(env)
    
    # Verify both filters are registered
    assert 'test_lower' in env.filters
    assert 'test_reverse' in env.filters
    
    # Test both filters work
    assert env.filters['test_lower']('HELLO') == 'hello'
    assert env.filters['test_reverse']('hello') == 'olleh'


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    
    # Store original directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Verify we start in original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager
        with work_in(temp_path):
            # Inside context, we should be in temp_dir
            assert os.getcwd() == str(temp_path)
        
        # After exiting context, we should be back in original directory
        assert os.getcwd() == original_dir


def test_work_in_no_dirname():
    """Test work_in context manager with None dirname."""
    import os
    
    original_dir = os.getcwd()
    
    # Use work_in with None (no directory change)
    with work_in(None):
        # Should still be in original directory
        assert os.getcwd() == original_dir
    
    # After exiting, should still be in original directory
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    import os
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                # Verify we're in temp_dir
                assert os.getcwd() == temp_dir
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be restored to original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in context manager with string path."""
    import os
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use string path instead of Path object
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Create a test filter function
    def uppercase(value):
        """Convert a string to uppercase."""
        return value.upper()
    
    # Wrap it with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test that multiple filters can be created independently."""
    def lowercase(value):
        return value.lower()
    
    def reverse(value):
        return value[::-1]
    
    # Create extensions for both filters
    LowercaseExt = simple_filter(lowercase)
    ReverseExt = simple_filter(reverse)
    
    # Create environment and apply extensions
    env = StrictEnvironment()
    LowercaseExt(env)
    ReverseExt(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test the filters work correctly
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


def test_simple_filter_with_complex_function():
    """Test simple_filter with a more complex filter function."""
    def repeat(value, times=2):
        return value * times
    
    RepeatExt = simple_filter(repeat)
    env = StrictEnvironment()
    RepeatExt(env)
    
    assert 'repeat' in env.filters
    assert env.filters['repeat']('ab', 3) == 'ababab'


# LLM-generated content at query #36
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    import os
    from pathlib import Path
    
    # Store the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager to change directory
        with work_in(temp_path):
            # Inside context, we should be in the temp directory
            assert os.getcwd() == str(temp_path)
        
        # After exiting context, we should be back in original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Using work_in with None should not change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Should still be in original directory after exit
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Verify exception doesn't prevent directory restoration
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be back in original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in context manager with string path."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use string path instead of Path object
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Should be back in original directory
        assert os.getcwd() == original_dir


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None argument doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                assert os.getcwd() == tmpdir1
            assert os.getcwd() == original_dir


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if exception occurs
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #39
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        
        # Test that we return to the original directory after exiting
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_none():
    """Test the work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_returns_to_original_on_exception():
    """Test that work_in returns to original directory even on exception."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        
        # Verify we're back in the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_path_object():
    """Test the work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_work_in(tmp_path):
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that we return to the original directory after exiting
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test the work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test that work_in restores original directory even on exception."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    # Test that directory is restored even if an exception occurs
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test the work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    # Test with Path object
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test the work_in context manager with string path."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    # Test with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #41
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #42
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a Jinja2 extension with filter."""
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase_filter)
    
    # Verify it's a class that extends Extension
    assert issubclass(extension_class, Extension)
    
    # Verify the extension name matches the function name
    assert extension_class.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and initialize the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    
    # Verify the filter works correctly
    assert env.filters['uppercase_filter']('hello') == 'HELLO'
    assert env.filters['uppercase_filter']('test') == 'TEST'
    
    # Test with another filter function
    def reverse_filter(value):
        """Reverse a string."""
        return value[::-1]
    
    extension_class2 = simple_filter(reverse_filter)
    assert extension_class2.__name__ == 'reverse_filter'
    
    env2 = StrictEnvironment()
    extension_instance2 = extension_class2(env2)
    
    assert 'reverse_filter' in env2.filters
    assert env2.filters['reverse_filter']('hello') == 'olleh'


# LLM-generated content at query #43
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a proper Jinja2 extension."""
    # Define a simple test filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a class
    assert isinstance(FilterExtension, type)
    
    # Verify the class name matches the filter function name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Create a Jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter actually works
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_custom_filter():
    """Test simple_filter with a custom filter function."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    result = env.filters['reverse_string']('cookiecutter')
    assert result == 'retteccikoo'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def lowercase(value):
        return value.lower()
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    LowercaseExt = simple_filter(lowercase)
    PrefixExt = simple_filter(add_prefix)
    
    env = StrictEnvironment()
    LowercaseExt(env)
    PrefixExt(env)
    
    assert 'lowercase' in env.filters
    assert 'add_prefix' in env.filters
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['add_prefix']('test') == 'prefix_test'


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_none():
    """Test work_in with None stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should restore original directory despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    test_dir = str(tmp_path / "test_subdir")
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify directory was restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #46
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a Jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a type
    assert isinstance(FilterExtension, type)
    
    # Verify it's a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and add the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    
    # Verify the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test multiple simple_filter decorators work independently."""
    def lowercase(value):
        return value.lower()
    
    def reverse(value):
        return value[::-1]
    
    LowercaseExtension = simple_filter(lowercase)
    ReverseExtension = simple_filter(reverse)
    
    env = StrictEnvironment(extensions=[LowercaseExtension, ReverseExtension])
    
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


def test_simple_filter_in_template():
    """Test simple_filter works within Jinja2 template rendering."""
    def double(value):
        return value * 2
    
    DoubleExtension = simple_filter(double)
    env = StrictEnvironment(extensions=[DoubleExtension])
    
    template = env.from_string('{{ text|double }}')
    result = template.render(text='ab')
    assert result == 'abab'


# LLM-generated content at query #47
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Decorate it with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create an environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test that simple_filter works with different filter functions."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    ReverseExtension = simple_filter(reverse_string)
    env = StrictEnvironment()
    ReverseExtension(env)
    
    assert 'reverse_string' in env.filters
    result = env.filters['reverse_string']('abc')
    assert result == 'cba'


def test_simple_filter_with_complex_filter():
    """Test simple_filter with a more complex filter function."""
    def multiply_by_two(value):
        """Multiply value by two."""
        return value * 2
    
    MultiplyExtension = simple_filter(multiply_by_two)
    env = StrictEnvironment()
    MultiplyExtension(env)
    
    assert 'multiply_by_two' in env.filters
    result = env.filters['multiply_by_two'](21)
    assert result == 42


# LLM-generated content at query #48
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert a string to uppercase."""
        return str(value).upper()

    # Create the extension using simple_filter
    extension_class = simple_filter(uppercase)

    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase'

    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)

    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)

    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] == uppercase

    # Verify the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'


def test_simple_filter_with_custom_function():
    """Test simple_filter with a different custom function."""
    def double(value):
        """Double a number."""
        return int(value) * 2

    extension_class = simple_filter(double)

    env = StrictEnvironment()
    extension_instance = extension_class(env)

    assert 'double' in env.filters
    assert env.filters['double'](5) == 10
    assert env.filters['double'](21) == 42


def test_simple_filter_extension_initialization():
    """Test that simple_filter extension initializes correctly with environment."""
    def reverse_string(value):
        """Reverse a string."""
        return str(value)[::-1]

    extension_class = simple_filter(reverse_string)
    env = StrictEnvironment()

    # The extension should be instantiable
    extension = extension_class(env)

    # The filter should be available in the environment
    assert env.filters['reverse_string']('abc') == 'cba'


# LLM-generated content at query #49
#--------------------------

```python
def test_simple_filter():
    """Test the simple_filter decorator function."""
    from jinja2 import Environment
    
    # Define a test filter function
    def test_upper(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(test_upper)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'test_upper'
    
    # Create a Jinja2 environment and instantiate the extension
    env = Environment(extensions=[FilterExtension])
    
    # Verify the filter was registered in the environment
    assert 'test_upper' in env.filters
    assert env.filters['test_upper'] is test_upper
    
    # Test that the filter works correctly
    template = env.from_string('{{ "hello" | test_upper }}')
    result = template.render()
    assert result == 'HELLO'


def test_simple_filter_with_different_filter():
    """Test simple_filter with a different filter function."""
    from jinja2 import Environment
    
    # Define another test filter function
    def reverse_string(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(reverse_string)
    
    # Verify the extension name
    assert FilterExtension.__name__ == 'reverse_string'
    
    # Create environment and test the filter
    env = Environment(extensions=[FilterExtension])
    assert 'reverse_string' in env.filters
    
    template = env.from_string('{{ "hello" | reverse_string }}')
    result = template.render()
    assert result == 'olleh'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    from jinja2 import Environment
    
    def to_upper(value):
        return str(value).upper()
    
    def to_lower(value):
        return str(value).lower()
    
    UpperExtension = simple_filter(to_upper)
    LowerExtension = simple_filter(to_lower)
    
    env = Environment(extensions=[UpperExtension, LowerExtension])
    
    assert 'to_upper' in env.filters
    assert 'to_lower' in env.filters
    
    template = env.from_string('{{ "Hello" | to_upper }} {{ "World" | to_lower }}')
    result = template.render()
    assert result == 'HELLO world'


# LLM-generated content at query #50
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension from a function."""
    # Define a simple filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    ext = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def lowercase(value):
        """Convert value to lowercase."""
        return value.lower()
    
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    # Create extension classes
    lowercase_ext = simple_filter(lowercase)
    reverse_ext = simple_filter(reverse_string)
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse_string' in env.filters
    
    # Test both filters work
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_with_complex_function():
    """Test simple_filter with a more complex filter function."""
    def custom_join(items, separator=', '):
        """Join items with a custom separator."""
        return separator.join(str(i) for i in items)
    
    extension_class = simple_filter(custom_join)
    env = StrictEnvironment()
    extension_class(env)
    
    # Verify the filter works with arguments
    assert 'custom_join' in env.filters
    result = env.filters['custom_join']([1, 2, 3], separator=' | ')
    assert result == '1 | 2 | 3'


# LLM-generated content at query #51
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension that registers a filter."""
    
    # Define a simple test filter function
    def test_upper_filter(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Create the extension using simple_filter
    extension_class = simple_filter(test_upper_filter)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'test_upper_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'test_upper_filter' in env.filters
    assert env.filters['test_upper_filter'] == test_upper_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|test_upper_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'
    
    # Test with another filter function
    def test_reverse_filter(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    extension_class_2 = simple_filter(test_reverse_filter)
    assert extension_class_2.__name__ == 'test_reverse_filter'
    
    env2 = StrictEnvironment()
    extension_instance_2 = extension_class_2(env2)
    
    assert 'test_reverse_filter' in env2.filters
    template2 = env2.from_string('{{ text|test_reverse_filter }}')
    result2 = template2.render(text='cookiecutter')
    assert result2 == 'rettecikook'


# LLM-generated content at query #52
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Wrap it with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'
    assert env.filters['uppercase']('test') == 'TEST'


def test_simple_filter_multiple_filters():
    """Test that multiple filters can be created independently."""
    def lowercase(value):
        """Convert value to lowercase."""
        return value.lower()
    
    def reverse(value):
        """Reverse the value."""
        return value[::-1]
    
    # Create extensions for both filters
    LowercaseExt = simple_filter(lowercase)
    ReverseExt = simple_filter(reverse)
    
    # Verify they have different names
    assert LowercaseExt.__name__ == 'lowercase'
    assert ReverseExt.__name__ == 'reverse'
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    LowercaseExt(env)
    ReverseExt(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test both filters work
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


def test_simple_filter_with_jinja_template():
    """Test that simple_filter works within jinja2 templates."""
    def add_prefix(value):
        """Add 'prefix_' to the value."""
        return f'prefix_{value}'
    
    PrefixExt = simple_filter(add_prefix)
    env = StrictEnvironment(extensions=[PrefixExt])
    
    # Test filter in a template
    template = env.from_string('{{ name|add_prefix }}')
    result = template.render(name='test')
    
    assert result == 'prefix_test'


# LLM-generated content at query #53
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter decorator creates a jinja2 extension with filters."""
    from jinja2 import Environment
    
    # Define a simple filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Decorate it with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a jinja2 environment and add the extension
    env = Environment(extensions=[FilterExtension])
    
    # Verify the filter is registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|uppercase }}')
    result = template.render(text='hello world')
    assert result == 'HELLO WORLD'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    from jinja2 import Environment
    
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    def add_prefix(value):
        """Add prefix to value."""
        return f"prefix_{value}"
    
    ReverseExtension = simple_filter(reverse_string)
    PrefixExtension = simple_filter(add_prefix)
    
    env = Environment(extensions=[ReverseExtension, PrefixExtension])
    
    assert 'reverse_string' in env.filters
    assert 'add_prefix' in env.filters
    
    template = env.from_string('{{ text|reverse_string|add_prefix }}')
    result = template.render(text='hello')
    assert result == 'prefix_olleh'


def test_simple_filter_with_arguments():
    """Test that simple_filter works with filter functions that take arguments."""
    from jinja2 import Environment
    
    def repeat(value, times=2):
        """Repeat value n times."""
        return value * times
    
    RepeatExtension = simple_filter(repeat)
    env = Environment(extensions=[RepeatExtension])
    
    assert 'repeat' in env.filters
    
    template = env.from_string('{{ text|repeat(3) }}')
    result = template.render(text='x')
    assert result == 'xxx'


# LLM-generated content at query #54
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a proper Jinja2 extension."""
    
    def my_custom_filter(value):
        """A simple test filter that converts value to uppercase."""
        return str(value).upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify the extension class is created
    assert FilterExtension is not None
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the filter function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter is registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly when used
    assert env.filters['my_custom_filter']('hello') == 'HELLO'
    assert env.filters['my_custom_filter']('test') == 'TEST'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    
    def reverse_string(value):
        """A filter that reverses a string."""
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    
    def filter_one(value):
        return str(value) + '_one'
    
    def filter_two(value):
        return str(value) + '_two'
    
    ExtOne = simple_filter(filter_one)
    ExtTwo = simple_filter(filter_two)
    
    env = StrictEnvironment()
    ext_one = ExtOne(env)
    ext_two = ExtTwo(env)
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    assert env.filters['filter_one']('test') == 'test_one'
    assert env.filters['filter_two']('test') == 'test_two'


# LLM-generated content at query #55
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Create the extension using simple_filter
    extension_class = simple_filter(uppercase)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    template = env.from_string('{{ name | uppercase }}')
    result = template.render(name='hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    assert extension_class.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_string' in env.filters
    
    template = env.from_string('{{ text | reverse_string }}')
    result = template.render(text='hello')
    assert result == 'olleh'


def test_simple_filter_extension_inheritance():
    """Test that simple_filter returns a proper Extension subclass."""
    def dummy_filter(value):
        return value
    
    extension_class = simple_filter(dummy_filter)
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Verify it can be instantiated
    env = StrictEnvironment()
    instance = extension_class(env)
    assert isinstance(instance, Extension)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify directory was restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    
    def my_custom_filter(value):
        """A simple test filter function."""
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(my_custom_filter)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'my_custom_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works
    assert env.filters['my_custom_filter']('hello') == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    assert extension_class.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_string' in env.filters
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_preserves_function_behavior():
    """Test that simple_filter preserves the original function behavior."""
    
    def multiply_by_two(value):
        """Multiply a number by two."""
        return value * 2
    
    extension_class = simple_filter(multiply_by_two)
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Test with different types
    assert env.filters['multiply_by_two'](5) == 10
    assert env.filters['multiply_by_two']([1, 2]) == [1, 2, 1, 2]


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a Jinja2 extension with filter function."""
    
    def my_custom_filter(value):
        """A simple filter that converts value to uppercase."""
        return str(value).upper()
    
    # Create the extension class
    extension_class = simple_filter(my_custom_filter)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'my_custom_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    
    # Verify the filter function works correctly
    assert env.filters['my_custom_filter']('hello') == 'HELLO'
    assert env.filters['my_custom_filter']('test') == 'TEST'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    
    def reverse_string(value):
        """A filter that reverses a string."""
        return str(value)[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    # Verify extension class name
    assert extension_class.__name__ == 'reverse_string'
    
    # Create environment and instantiate extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify filter is registered
    assert 'reverse_string' in env.filters
    
    # Verify filter functionality
    assert env.filters['reverse_string']('hello') == 'olleh'
    assert env.filters['reverse_string']('abc') == 'cba'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    
    def double(value):
        """Filter that doubles a number."""
        return int(value) * 2
    
    def triple(value):
        """Filter that triples a number."""
        return int(value) * 3
    
    double_ext = simple_filter(double)
    triple_ext = simple_filter(triple)
    
    env = StrictEnvironment()
    double_ext(env)
    triple_ext(env)
    
    # Both filters should be registered
    assert 'double' in env.filters
    assert 'triple' in env.filters
    
    # Both should work independently
    assert env.filters['double'](5) == 10
    assert env.filters['triple'](5) == 15


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in restores directory even if exception is raised
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test that work_in works with Path objects
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in with None argument doesn't change directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    with contextlib.suppress(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    
    # Should restore original directory despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string path."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path object."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_workdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to change into
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if an exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still restore to original directory
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in with None argument keeps current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in accepts Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_none():
    """Test work_in with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                assert os.getcwd() == tmpdir1
            assert os.getcwd() == original_dir


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if exception occurs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore to original directory despite the exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #10
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a valid Jinja2 extension."""
    
    def my_test_filter(value):
        """Test filter that converts to uppercase."""
        return str(value).upper()
    
    # Create the extension class
    FilterExtension = simple_filter(my_test_filter)
    
    # Verify the extension class name matches the function name
    assert FilterExtension.__name__ == 'my_test_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Create a Jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_test_filter' in env.filters
    assert env.filters['my_test_filter'] is my_test_filter
    
    # Test that the filter works correctly
    result = env.filters['my_test_filter']('hello')
    assert result == 'HELLO'
    
    # Test with a template
    template = env.from_string('{{ value|my_test_filter }}')
    output = template.render(value='test')
    assert output == 'TEST'


# LLM-generated content at query #11
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter correctly wraps a function as a Jinja2 extension."""
    # Define a simple test filter function
    def uppercase_filter(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Create the extension class using simple_filter
    FilterExtension = simple_filter(uppercase_filter)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and initialize the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'
    
    # Test with another filter function
    def reverse_filter(value):
        """Reverse the value."""
        return str(value)[::-1]
    
    ReverseExtension = simple_filter(reverse_filter)
    env2 = StrictEnvironment()
    ReverseExtension(env2)
    
    assert 'reverse_filter' in env2.filters
    assert env2.filters['reverse_filter']('abc') == 'cba'


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir1):
            assert os.getcwd() == temp_dir1
            
            with work_in(temp_dir2):
                assert os.getcwd() == temp_dir2
            
            assert os.getcwd() == temp_dir1
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir1)
        rmtree(temp_dir2)


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to change into
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test work_in with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still restore the original directory
        assert os.getcwd() == original_dir


# LLM-generated content at query #14
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Wrap it with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] == uppercase
    
    # Test that the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple filters can be created independently."""
    def lowercase(value):
        return value.lower()
    
    def reverse_string(value):
        return value[::-1]
    
    LowercaseExtension = simple_filter(lowercase)
    ReverseExtension = simple_filter(reverse_string)
    
    env = StrictEnvironment()
    LowercaseExtension(env)
    ReverseExtension(env)
    
    assert 'lowercase' in env.filters
    assert 'reverse_string' in env.filters
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_with_parameters():
    """Test that simple_filter works with functions that take parameters."""
    def repeat(value, times=2):
        return value * times
    
    RepeatExtension = simple_filter(repeat)
    env = StrictEnvironment()
    RepeatExtension(env)
    
    assert env.filters['repeat']('x', 3) == 'xxx'
    assert env.filters['repeat']('ab', 2) == 'abab'


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in context manager with None dirname stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores original directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should restore original directory even after exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in context manager with string path."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in with None argument doesn't change directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with contextlib.suppress(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string path argument."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path object argument."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test changing to a different directory
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        
        # Test that we return to the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_none():
    """Test the work_in context manager with None (no directory change)."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores original directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        with contextlib.suppress(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        
        # Should still be back in original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


def test_work_in_with_path_object():
    """Test the work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with work_in(tmp_path):
            assert os.getcwd() == str(tmp_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    import os
    from pathlib import Path
    
    # Get the initial working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test entering a different directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that we return to the original directory
        assert os.getcwd() == original_cwd
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_cwd
    
    assert os.getcwd() == original_cwd
    
    # Test that we return even if an exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be back in original directory
        assert os.getcwd() == original_cwd
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_cwd


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    import tempfile
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test with None as dirname (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that original directory is restored even if an exception occurs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be back in original directory despite the exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager to change directory
        with work_in(temp_dir):
            # Inside context, should be in temp_dir
            assert os.getcwd() == temp_dir
        
        # After exiting context, should be back in original directory
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_work_in_with_none():
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        # Inside context with None, should still be in original directory
        assert os.getcwd() == original_dir
    
    # After exiting context, should still be in original directory
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
            raise ValueError("Test exception")
    except ValueError:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Should be back in original directory even after exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_cwd = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory after exit
    assert os.getcwd() == original_cwd


def test_work_in_none(tmp_path):
    """Test work_in with None argument stays in current directory."""
    original_cwd = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_cwd
    
    assert os.getcwd() == original_cwd


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception is raised."""
    original_cwd = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify directory was restored despite exception
    assert os.getcwd() == original_cwd


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string path argument."""
    original_cwd = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_cwd


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path object argument."""
    original_cwd = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in context manager works with Path objects."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import tempfile
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Enter the context manager with a new directory
        with work_in(tmpdir):
            # Inside context, we should be in tmpdir
            assert os.getcwd() == tmpdir
        
        # After exiting, we should be back in original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that we return even if an exception occurs
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should still be in original directory after exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in with None argument stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Directory should be restored despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None stays in current directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that original directory is restored
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that directory is restored even if exception is raised
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify original directory is restored after exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in context manager with string path."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with string path instead of Path object
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    # Test that we're still in the original directory after exiting
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test that work_in restores directory even if exception is raised
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify directory was restored despite the exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Test with Path object
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    test_dir = str(tmp_path / "test_subdir")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test with string path
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #29
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return str(value).upper()

    # Create the extension using simple_filter decorator
    extension_class = simple_filter(uppercase_filter)

    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase_filter'

    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)

    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)

    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] == uppercase_filter

    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators create independent extensions."""
    def lowercase_filter(value):
        """Convert string to lowercase."""
        return str(value).lower()

    def reverse_filter(value):
        """Reverse a string."""
        return str(value)[::-1]

    # Create extensions for both filters
    lowercase_ext = simple_filter(lowercase_filter)
    reverse_ext = simple_filter(reverse_filter)

    # Create environments and register extensions
    env1 = StrictEnvironment()
    env2 = StrictEnvironment()

    lowercase_ext(env1)
    reverse_ext(env2)

    # Verify each environment has the correct filter
    assert 'lowercase_filter' in env1.filters
    assert 'reverse_filter' in env2.filters
    assert 'reverse_filter' not in env1.filters
    assert 'lowercase_filter' not in env2.filters

    # Test filter functionality
    assert env1.filters['lowercase_filter']('HELLO') == 'hello'
    assert env2.filters['reverse_filter']('hello') == 'olleh'


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            temp_path1 = Path(temp_dir1)
            temp_path2 = Path(temp_dir2)
            
            with work_in(temp_path1):
                assert os.getcwd() == str(temp_path1)
                
                with work_in(temp_path2):
                    assert os.getcwd() == str(temp_path2)
                
                assert os.getcwd() == str(temp_path1)
            
            assert os.getcwd() == original_dir


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    import os
    from pathlib import Path
    import tempfile
    
    # Get the current working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_cwd
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_cwd
    
    assert os.getcwd() == original_cwd


def test_work_in_with_path_object():
    """Test the work_in context manager with Path object."""
    import os
    from pathlib import Path
    import tempfile
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_cwd


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    import os
    import tempfile
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original directory even after exception
        assert os.getcwd() == original_cwd


# LLM-generated content at query #32
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension with filter."""
    def my_custom_filter(value):
        """A simple test filter that uppercases input."""
        return str(value).upper()
    
    # Create the extension class
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Create a jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] == my_custom_filter
    
    # Verify the filter works correctly
    template = env.from_string("{{ value | my_custom_filter }}")
    result = template.render(value='hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators create independent extensions."""
    def filter_one(value):
        return value + '_one'
    
    def filter_two(value):
        return value + '_two'
    
    # Create two extensions
    FilterExtensionOne = simple_filter(filter_one)
    FilterExtensionTwo = simple_filter(filter_two)
    
    # Verify they have correct names
    assert FilterExtensionOne.__name__ == 'filter_one'
    assert FilterExtensionTwo.__name__ == 'filter_two'
    
    # Create environment and register both
    env = StrictEnvironment()
    FilterExtensionOne(env)
    FilterExtensionTwo(env)
    
    # Verify both filters are registered and work
    template = env.from_string("{{ value | filter_one | filter_two }}")
    result = template.render(value='test')
    assert result == 'test_one_two'


def test_simple_filter_with_complex_function():
    """Test simple_filter with a more complex filter function."""
    def reverse_string(value):
        """Reverses a string."""
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    env = StrictEnvironment()
    FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    template = env.from_string("{{ text | reverse_string }}")
    result = template.render(text='cookiecutter')
    assert result == 'retteceikooc'


# LLM-generated content at query #33
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    import os
    import tempfile
    from pathlib import Path
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that we're back to the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    import os
    import tempfile
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    import os
    import tempfile
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore to original directory even after exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that we're back to the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that work_in restores directory even if an exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be back to original directory despite exception
        assert os.getcwd() == original_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a Jinja2 extension."""
    from jinja2 import Environment
    
    # Define a simple filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Apply the decorator
    FilterExtension = simple_filter(uppercase)
    
    # Create a Jinja2 environment and add the extension
    env = Environment(extensions=[FilterExtension])
    
    # Verify the filter was registered
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works
    template = env.from_string('{{ text | uppercase }}')
    result = template.render(text='hello')
    assert result == 'HELLO'
    
    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == 'uppercase'


def test_simple_filter_multiple_filters():
    """Test that multiple filters can be created independently."""
    from jinja2 import Environment
    
    def lowercase(value):
        """Convert string to lowercase."""
        return value.lower()
    
    def reverse(value):
        """Reverse a string."""
        return value[::-1]
    
    LowercaseExtension = simple_filter(lowercase)
    ReverseExtension = simple_filter(reverse)
    
    env = Environment(extensions=[LowercaseExtension, ReverseExtension])
    
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    template = env.from_string('{{ text | lowercase | reverse }}')
    result = template.render(text='HELLO')
    assert result == 'olleh'


def test_simple_filter_with_arguments():
    """Test that simple_filter works with filter functions that take arguments."""
    from jinja2 import Environment
    
    def repeat(value, times=2):
        """Repeat a string multiple times."""
        return value * times
    
    RepeatExtension = simple_filter(repeat)
    env = Environment(extensions=[RepeatExtension])
    
    template = env.from_string('{{ text | repeat(3) }}')
    result = template.render(text='ab')
    assert result == 'ababab'


# LLM-generated content at query #36
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension with filter."""
    def my_custom_filter(value):
        """Test filter that uppercases a string."""
        return str(value).upper()
    
    # Create the extension class
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify the extension class name matches the function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly
    assert env.filters['my_custom_filter']('hello') == 'HELLO'
    assert env.filters['my_custom_filter']('test') == 'TEST'


def test_simple_filter_with_multiple_filters():
    """Test that multiple simple filters can be created independently."""
    def filter_one(value):
        return str(value).lower()
    
    def filter_two(value):
        return str(value)[::-1]
    
    # Create extension classes for both filters
    FilterOne = simple_filter(filter_one)
    FilterTwo = simple_filter(filter_two)
    
    # Verify names
    assert FilterOne.__name__ == 'filter_one'
    assert FilterTwo.__name__ == 'filter_two'
    
    # Create environment and register both extensions
    env = StrictEnvironment()
    FilterOne(env)
    FilterTwo(env)
    
    # Verify both filters are registered
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    
    # Test both filters work
    assert env.filters['filter_one']('HELLO') == 'hello'
    assert env.filters['filter_two']('hello') == 'olleh'


def test_simple_filter_with_complex_function():
    """Test simple_filter with a more complex filter function."""
    def word_count(value):
        """Count words in a string."""
        return len(str(value).split())
    
    FilterExtension = simple_filter(word_count)
    env = StrictEnvironment()
    FilterExtension(env)
    
    assert env.filters['word_count']('hello world test') == 3
    assert env.filters['word_count']('single') == 1
    assert env.filters['word_count']('') == 1  # split on empty string returns ['']


# LLM-generated content at query #37
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension from a filter function."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return str(value).upper()
    
    # Create the extension class
    extension_class = simple_filter(uppercase)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators work independently."""
    def lowercase(value):
        """Convert string to lowercase."""
        return str(value).lower()
    
    def reverse(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    # Create extension classes for both filters
    lowercase_ext = simple_filter(lowercase)
    reverse_ext = simple_filter(reverse)
    
    # Verify names are correct
    assert lowercase_ext.__name__ == 'lowercase'
    assert reverse_ext.__name__ == 'reverse'
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test both filters work
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


def test_simple_filter_with_arguments():
    """Test that simple_filter works with filters that take arguments."""
    def repeat(value, times=2):
        """Repeat a string n times."""
        return str(value) * times
    
    extension_class = simple_filter(repeat)
    env = StrictEnvironment()
    extension_class(env)
    
    assert 'repeat' in env.filters
    assert env.filters['repeat']('x') == 'xx'
    assert env.filters['repeat']('x', 3) == 'xxx'


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that we're back in the original directory after exiting
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in with None argument doesn't change directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even if exception is raised."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be back in original directory despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #39
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    import tempfile
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Verify we start in the original directory
        assert os.getcwd() == original_dir
        
        # Use work_in context manager
        with work_in(tmpdir):
            # Inside context, we should be in tmpdir
            assert os.getcwd() == tmpdir
        
        # After exiting context, we should be back in original directory
        assert os.getcwd() == original_dir
    
    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir
    
    # Test that it restores even if an exception occurs
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be restored to original directory
        assert os.getcwd() == original_dir
    
    # Test with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #40
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter decorator creates a jinja2 extension with filter."""
    # Define a test filter function
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Create the extension using simple_filter decorator
    extension_class = simple_filter(uppercase_filter)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] == uppercase_filter
    
    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_filter(value):
        """Reverse a string."""
        return value[::-1]
    
    extension_class = simple_filter(reverse_filter)
    
    assert extension_class.__name__ == 'reverse_filter'
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_filter' in env.filters
    result = env.filters['reverse_filter']('hello')
    assert result == 'olleh'


def test_simple_filter_extension_inheritance():
    """Test that simple_filter creates a proper Extension subclass."""
    def test_filter(value):
        return str(value)
    
    extension_class = simple_filter(test_filter)
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Verify it can be instantiated with an environment
    env = StrictEnvironment()
    instance = extension_class(env)
    assert isinstance(instance, Extension)


# LLM-generated content at query #41
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    def my_custom_filter(value):
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(my_custom_filter)
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Verify the extension name matches the filter function name
    assert extension_class.__name__ == 'my_custom_filter'
    
    # Create an environment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly
    assert env.filters['my_custom_filter']('hello') == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    assert extension_class.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_string' in env.filters
    assert env.filters['reverse_string']('abc') == 'cba'


def test_simple_filter_multiple_extensions():
    """Test creating multiple simple_filter extensions."""
    def filter_one(value):
        return value + '_one'
    
    def filter_two(value):
        return value + '_two'
    
    ext_class_one = simple_filter(filter_one)
    ext_class_two = simple_filter(filter_two)
    
    env = StrictEnvironment()
    ext_class_one(env)
    ext_class_two(env)
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    assert env.filters['filter_one']('test') == 'test_one'
    assert env.filters['filter_two']('test') == 'test_two'


# LLM-generated content at query #42
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    def my_test_filter(value):
        """Test filter that converts to uppercase."""
        return str(value).upper()
    
    # Create the extension using simple_filter decorator
    extension_class = simple_filter(my_test_filter)
    
    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'my_test_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'my_test_filter' in env.filters
    assert env.filters['my_test_filter'] is my_test_filter
    
    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | my_test_filter }}")
    result = template.render()
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Test filter that reverses a string."""
        return str(value)[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    assert extension_class.__name__ == 'reverse_string'
    assert issubclass(extension_class, Extension)
    
    env = StrictEnvironment()
    extension = extension_class(env)
    
    assert 'reverse_string' in env.filters
    template = env.from_string("{{ 'test' | reverse_string }}")
    result = template.render()
    assert result == 'tset'


def test_simple_filter_preserves_filter_function():
    """Test that simple_filter preserves the original filter function."""
    def multiply_by_two(value):
        """Test filter that multiplies by 2."""
        return int(value) * 2
    
    extension_class = simple_filter(multiply_by_two)
    env = StrictEnvironment()
    extension = extension_class(env)
    
    # The registered filter should be the original function
    registered_filter = env.filters['multiply_by_two']
    assert registered_filter(5) == 10
    assert registered_filter(3) == 6


# LLM-generated content at query #43
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path_obj = Path(tmpdir)
        with work_in(path_obj):
            assert os.getcwd() == str(path_obj)
        
        assert os.getcwd() == original_dir


def test_work_in_default_none():
    """Test work_in context manager with default None parameter."""
    original_dir = os.getcwd()
    
    # Test calling work_in without arguments
    with work_in():
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes directory and restores it."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Test that directory is restored even after exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                
                assert os.getcwd() == tmpdir1
            
            assert os.getcwd() == original_dir


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test that work_in with None argument doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify directory was restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test that work_in works with Path objects."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with work_in(tmpdir_path):
            assert os.getcwd() == str(tmpdir_path)
        
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test that work_in works with string paths."""
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #46
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test changing directory
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        
        # Test that we return to original directory
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


def test_work_in_none():
    """Test the work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test with None - should stay in current directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_returns_to_original_on_exception():
    """Test that work_in returns to original directory even on exception."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        
        # Should return to original directory despite exception
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


def test_work_in_with_path_object():
    """Test the work_in context manager with Path object."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    test_path = Path(test_dir)
    
    try:
        with work_in(test_path):
            assert os.getcwd() == test_dir
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


# LLM-generated content at query #47
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even on exception."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in restores directory even when exception occurs
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_work_in(tmp_path):
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    # Test that work_in changes to the specified directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Test that work_in restores the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none(tmp_path):
    """Test work_in with None argument doesn't change directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test work_in restores original directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should restore original directory despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test work_in works with string path argument."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test work_in works with Path object argument."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    import os
    from pathlib import Path
    
    # Get the original working directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to change into
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that work_in changes to the specified directory
        with work_in(temp_path):
            assert os.getcwd() == str(temp_path)
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception is raised."""
    import os
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test that directory is restored even if exception occurs
        try:
            with work_in(temp_path):
                assert os.getcwd() == str(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify directory was restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_string_path():
    """Test work_in works with string paths."""
    import os
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with string path instead of Path object
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        assert os.getcwd() == original_dir


# LLM-generated content at query #50
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores working directory."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in changes to the specified directory
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # Test that work_in restores the original directory after exiting
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    # Test that work_in with None doesn't change directory
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in context manager restores directory even when exception occurs."""
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test that work_in restores directory even if exception is raised
        with pytest.raises(ValueError):
            with work_in(temp_dir):
                assert os.getcwd() == temp_dir
                raise ValueError("Test exception")
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


def test_work_in_with_path_object():
    """Test work_in context manager with Path object."""
    original_dir = os.getcwd()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(temp_dir):
            assert os.getcwd() == str(temp_dir)
        
        assert os.getcwd() == original_dir
    finally:
        rmtree(temp_dir)


# LLM-generated content at query #51
#--------------------------

```python
def test_work_in():
    """Test work_in context manager changes and restores directory."""
    import os
    from pathlib import Path
    
    # Store original directory
    original_dir = os.getcwd()
    
    # Create a temporary directory to work in
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that work_in changes to the specified directory
        with work_in(tmpdir):
            assert os.getcwd() == tmpdir
        
        # Test that work_in restores the original directory
        assert os.getcwd() == original_dir
    
    # Test work_in with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test work_in restores directory even when exception occurs."""
    import os
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test that directory is restored even if exception occurs
        try:
            with work_in(tmpdir):
                assert os.getcwd() == tmpdir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should be restored despite exception
        assert os.getcwd() == original_dir


def test_work_in_with_path_object():
    """Test work_in works with Path objects."""
    import os
    from pathlib import Path
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        with work_in(tmppath):
            assert os.getcwd() == str(tmppath)
        
        assert os.getcwd() == original_dir


def test_work_in_nested():
    """Test nested work_in context managers."""
    import os
    
    original_dir = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with work_in(tmpdir1):
                assert os.getcwd() == tmpdir1
                
                with work_in(tmpdir2):
                    assert os.getcwd() == tmpdir2
                
                assert os.getcwd() == tmpdir1
            
            assert os.getcwd() == original_dir


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in():
    """Test the work_in context manager."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test entering and exiting context manager
        assert os.getcwd() == original_dir
        
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


def test_work_in_with_none():
    """Test work_in context manager with None argument."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception():
    """Test that work_in restores directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


def test_work_in_with_path_object():
    """Test work_in with Path object."""
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir, onerror=force_delete)


# LLM-generated content at query #53
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a Jinja2 extension with filter."""
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Create the extension using simple_filter
    filter_extension = simple_filter(uppercase_filter)
    
    # Verify it returns a type that is a subclass of Extension
    assert issubclass(filter_extension, Extension)
    
    # Verify the extension class has the correct name
    assert filter_extension.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = filter_extension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] == uppercase_filter
    
    # Test that the filter works correctly
    result = env.filters['uppercase_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test simple_filter works with multiple different filters."""
    def lowercase_filter(value):
        """Convert string to lowercase."""
        return value.lower()
    
    def reverse_filter(value):
        """Reverse a string."""
        return value[::-1]
    
    # Create extensions for both filters
    lowercase_ext = simple_filter(lowercase_filter)
    reverse_ext = simple_filter(reverse_filter)
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)
    
    # Verify both filters are registered
    assert 'lowercase_filter' in env.filters
    assert 'reverse_filter' in env.filters
    
    # Test both filters work independently
    assert env.filters['lowercase_filter']('HELLO') == 'hello'
    assert env.filters['reverse_filter']('hello') == 'olleh'


def test_simple_filter_with_arguments():
    """Test simple_filter works with functions that take multiple arguments."""
    def repeat_filter(value, times=2):
        """Repeat a string multiple times."""
        return value * times
    
    filter_extension = simple_filter(repeat_filter)
    env = StrictEnvironment()
    filter_extension(env)
    
    # Verify filter is registered
    assert 'repeat_filter' in env.filters
    
    # Test the filter with default argument
    assert env.filters['repeat_filter']('a') == 'aa'
    
    # Test the filter with custom argument
    assert env.filters['repeat_filter']('a', 3) == 'aaa'


# LLM-generated content at query #54
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter creates a jinja2 extension with filter."""
    def my_test_filter(value):
        """Test filter that converts to uppercase."""
        return value.upper()
    
    # Create the extension class
    extension_class = simple_filter(my_test_filter)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'my_test_filter'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'my_test_filter' in env.filters
    assert env.filters['my_test_filter'] is my_test_filter
    
    # Test that the filter works correctly
    assert env.filters['my_test_filter']('hello') == 'HELLO'
    assert env.filters['my_test_filter']('world') == 'WORLD'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Test filter that reverses a string."""
        return value[::-1]
    
    # Create the extension class
    extension_class = simple_filter(reverse_string)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'reverse_string'
    
    # Create environment and instantiate extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter works
    assert env.filters['reverse_string']('abc') == 'cba'
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    def filter1(value):
        return value.upper()
    
    def filter2(value):
        return value.lower()
    
    ext_class1 = simple_filter(filter1)
    ext_class2 = simple_filter(filter2)
    
    env = StrictEnvironment()
    ext_class1(env)
    ext_class2(env)
    
    # Both filters should be registered
    assert 'filter1' in env.filters
    assert 'filter2' in env.filters
    assert env.filters['filter1']('Hello') == 'HELLO'
    assert env.filters['filter2']('Hello') == 'hello'


# LLM-generated content at query #55
#--------------------------

```python
def test_work_in(tmp_path):
    """Test that work_in context manager changes and restores directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    # Verify we start in the original directory
    assert os.getcwd() == original_dir
    
    # Use work_in to change to test directory
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    # Verify we're back in the original directory
    assert os.getcwd() == original_dir


def test_work_in_with_none():
    """Test that work_in with None doesn't change directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_on_exception(tmp_path):
    """Test that work_in restores directory even when exception is raised."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify we're back in the original directory despite exception
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test that work_in works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test that work_in works with string paths."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #56
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a Jinja2 extension with filter."""
    def my_custom_filter(value):
        """A simple test filter that reverses a string."""
        return value[::-1]
    
    # Create the extension class
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Create a Jinja2 environment and initialize the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text | my_custom_filter }}')
    result = template.render(text='hello')
    assert result == 'olleh'


def test_simple_filter_with_different_function():
    """Test simple_filter with different filter functions."""
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return value.upper()
    
    FilterExtension = simple_filter(uppercase_filter)
    
    assert FilterExtension.__name__ == 'uppercase_filter'
    
    env = StrictEnvironment(extensions=[FilterExtension])
    
    assert 'uppercase_filter' in env.filters
    
    template = env.from_string('{{ text | uppercase_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test registering multiple simple_filter extensions."""
    def filter_one(value):
        return value + '_one'
    
    def filter_two(value):
        return value + '_two'
    
    Extension1 = simple_filter(filter_one)
    Extension2 = simple_filter(filter_two)
    
    env = StrictEnvironment(extensions=[Extension1, Extension2])
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    
    template = env.from_string('{{ text | filter_one | filter_two }}')
    result = template.render(text='test')
    assert result == 'test_one_two'


# LLM-generated content at query #57
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a simple test filter function
    def uppercase(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Create the extension using simple_filter
    extension_class = simple_filter(uppercase)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] == uppercase
    
    # Test that the filter works correctly
    assert env.filters['uppercase']('hello') == 'HELLO'
    assert env.filters['uppercase']('test123') == 'TEST123'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    def lowercase(value):
        """Convert value to lowercase."""
        return str(value).lower()
    
    def reverse(value):
        """Reverse the value."""
        return str(value)[::-1]
    
    # Create extensions for both filters
    lowercase_ext = simple_filter(lowercase)
    reverse_ext = simple_filter(reverse)
    
    # Create environment and register both extensions
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test both filters work independently
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


# LLM-generated content at query #58
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    
    # Define a test filter function
    def uppercase(value):
        """Convert a string to uppercase."""
        return str(value).upper()
    
    # Create the extension using simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    
    def lowercase(value):
        """Convert a string to lowercase."""
        return str(value).lower()
    
    def reverse(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    # Create multiple extensions
    LowercaseExtension = simple_filter(lowercase)
    ReverseExtension = simple_filter(reverse)
    
    env = StrictEnvironment()
    LowercaseExtension(env)
    ReverseExtension(env)
    
    # Verify both filters are registered
    assert 'lowercase' in env.filters
    assert 'reverse' in env.filters
    
    # Test that both filters work
    assert env.filters['lowercase']('HELLO') == 'hello'
    assert env.filters['reverse']('hello') == 'olleh'


def test_simple_filter_preserves_function_behavior():
    """Test that simple_filter preserves the original function behavior."""
    
    def add_prefix(value, prefix='PREFIX_'):
        """Add a prefix to a value."""
        return prefix + str(value)
    
    FilterExtension = simple_filter(add_prefix)
    env = StrictEnvironment()
    FilterExtension(env)
    
    # Test the filter with default argument
    result = env.filters['add_prefix']('test')
    assert result == 'PREFIX_test'
    
    # Test the filter with custom argument
    result = env.filters['add_prefix']('test', prefix='CUSTOM_')
    assert result == 'CUSTOM_test'


# LLM-generated content at query #59
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def test_upper(value):
        """Convert value to uppercase."""
        return value.upper()
    
    # Decorate the function with simple_filter
    filter_extension = simple_filter(test_upper)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(filter_extension, Extension)
    
    # Verify the extension class name matches the function name
    assert filter_extension.__name__ == 'test_upper'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = filter_extension(env)
    
    # Verify the filter was registered in the environment
    assert 'test_upper' in env.filters
    assert env.filters['test_upper'] is test_upper
    
    # Verify the filter works correctly
    assert env.filters['test_upper']('hello') == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    filter_extension = simple_filter(reverse_string)
    
    assert filter_extension.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    filter_extension(env)
    
    assert 'reverse_string' in env.filters
    assert env.filters['reverse_string']('hello') == 'olleh'


def test_simple_filter_with_lambda():
    """Test simple_filter with a lambda function."""
    test_func = lambda x: x * 2
    test_func.__name__ = 'double'
    
    filter_extension = simple_filter(test_func)
    
    assert filter_extension.__name__ == 'double'
    
    env = StrictEnvironment()
    filter_extension(env)
    
    assert 'double' in env.filters
    assert env.filters['double'](5) == 10


# LLM-generated content at query #60
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    def my_custom_filter(value):
        """A simple test filter that uppercases a string."""
        return str(value).upper()
    
    # Decorate the function
    FilterExtension = simple_filter(my_custom_filter)
    
    # Verify it returns a class
    assert isinstance(FilterExtension, type)
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Create a Jinja2 environment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] is my_custom_filter
    
    # Test that the filter works correctly
    result = env.filters['my_custom_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Reverses a string."""
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    result = env.filters['reverse_string']('abc')
    assert result == 'cba'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def add_prefix(value):
        return f'prefix_{value}'
    
    def add_suffix(value):
        return f'{value}_suffix'
    
    Extension1 = simple_filter(add_prefix)
    Extension2 = simple_filter(add_suffix)
    
    env = StrictEnvironment()
    ext1 = Extension1(env)
    ext2 = Extension2(env)
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #61
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter correctly wraps a function as a Jinja2 extension."""
    # Define a simple test filter function
    def uppercase_filter(value):
        """Convert a string to uppercase."""
        return str(value).upper()
    
    # Create the extension using simple_filter
    filter_extension = simple_filter(uppercase_filter)
    
    # Verify it returns a class that is a subclass of Extension
    assert issubclass(filter_extension, Extension)
    
    # Verify the extension class has the correct name
    assert filter_extension.__name__ == 'uppercase_filter'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = filter_extension(env)
    
    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter
    
    # Test that the filter works correctly
    assert env.filters['uppercase_filter']('hello') == 'HELLO'
    assert env.filters['uppercase_filter']('test') == 'TEST'


def test_simple_filter_with_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def reverse_filter(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    def double_filter(value):
        """Double a numeric value."""
        return int(value) * 2
    
    # Create extensions for both filters
    reverse_ext = simple_filter(reverse_filter)
    double_ext = simple_filter(double_filter)
    
    # Create environment and add both extensions
    env = StrictEnvironment()
    reverse_ext(env)
    double_ext(env)
    
    # Verify both filters are registered
    assert 'reverse_filter' in env.filters
    assert 'double_filter' in env.filters
    
    # Test both filters work independently
    assert env.filters['reverse_filter']('hello') == 'olleh'
    assert env.filters['double_filter'](5) == 10


def test_simple_filter_preserves_function_behavior():
    """Test that simple_filter preserves the original function behavior."""
    def custom_filter(value, suffix='!'):
        """Add a suffix to a string."""
        return str(value) + suffix
    
    filter_ext = simple_filter(custom_filter)
    env = StrictEnvironment()
    filter_ext(env)
    
    # Test that the filter function works as expected
    assert env.filters['custom_filter']('hello') == 'hello!'
    assert env.filters['custom_filter']('test', suffix='?') == 'test?'


# LLM-generated content at query #62
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    def test_func(value):
        """Test filter function that doubles a value."""
        return value * 2
    
    # Create the extension class
    extension_class = simple_filter(test_func)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'test_func'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'test_func' in env.filters
    assert env.filters['test_func'] is test_func
    
    # Verify the filter works correctly
    assert env.filters['test_func'](5) == 10
    assert env.filters['test_func']('ab') == 'abab'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators create independent extensions."""
    def filter_one(value):
        return value + 1
    
    def filter_two(value):
        return value * 3
    
    ext_one = simple_filter(filter_one)
    ext_two = simple_filter(filter_two)
    
    # Verify each has the correct name
    assert ext_one.__name__ == 'filter_one'
    assert ext_two.__name__ == 'filter_two'
    
    # Create environment and register both extensions
    env = StrictEnvironment()
    ext_one(env)
    ext_two(env)
    
    # Verify both filters are registered
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    
    # Verify they work independently
    assert env.filters['filter_one'](5) == 6
    assert env.filters['filter_two'](5) == 15


# LLM-generated content at query #63
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a jinja2 extension."""
    # Define a test filter function
    def test_uppercase(value):
        return value.upper()
    
    # Apply the simple_filter decorator
    extension_class = simple_filter(test_uppercase)
    
    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'test_uppercase'
    
    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    # Verify the filter was registered in the environment
    assert 'test_uppercase' in env.filters
    assert env.filters['test_uppercase'] is test_uppercase
    
    # Verify the filter works correctly
    result = env.filters['test_uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    
    assert extension_class.__name__ == 'reverse_string'
    
    env = StrictEnvironment()
    extension_instance = extension_class(env)
    
    assert 'reverse_string' in env.filters
    result = env.filters['reverse_string']('cookiecutter')
    assert result == 'rettecikoc'


def test_simple_filter_multiple_extensions():
    """Test creating multiple simple_filter extensions."""
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    ext1 = simple_filter(add_prefix)
    ext2 = simple_filter(add_suffix)
    
    env = StrictEnvironment()
    ext1(env)
    ext2(env)
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


# LLM-generated content at query #64
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return value.upper()
    
    # Decorate the function with simple_filter
    FilterExtension = simple_filter(uppercase)
    
    # Verify it returns a type that is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a jinja2 environment and initialize the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter is registered in the environment
    assert 'uppercase' in env.filters
    
    # Verify the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    """Test simple_filter with a different filter function."""
    def reverse_string(value):
        """Reverse a string."""
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == 'reverse_string'
    assert issubclass(FilterExtension, Extension)
    
    env = StrictEnvironment(extensions=[FilterExtension])
    assert 'reverse_string' in env.filters
    
    result = env.filters['reverse_string']('hello')
    assert result == 'olleh'


def test_simple_filter_extension_initialization():
    """Test that SimpleFilterExtension properly initializes with environment."""
    def test_filter(value):
        """Test filter function."""
        return f"filtered_{value}"
    
    FilterExtension = simple_filter(test_filter)
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter can be used in a template
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == 'filtered_test'


# LLM-generated content at query #65
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    def my_custom_filter(value):
        """A simple test filter that converts to uppercase."""
        return str(value).upper()
    
    # Apply the decorator
    FilterExtension = simple_filter(my_custom_filter)
    
    # Create a StrictEnvironment and initialize the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the extension was created with the correct name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] == my_custom_filter
    
    # Test that the filter works correctly
    template = env.from_string('{{ text|my_custom_filter }}')
    result = template.render(text='hello')
    assert result == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def filter_one(value):
        return str(value) + '_one'
    
    def filter_two(value):
        return str(value) + '_two'
    
    ExtensionOne = simple_filter(filter_one)
    ExtensionTwo = simple_filter(filter_two)
    
    env = StrictEnvironment()
    ExtensionOne(env)
    ExtensionTwo(env)
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    
    template = env.from_string('{{ text|filter_one|filter_two }}')
    result = template.render(text='test')
    assert result == 'test_one_two'


def test_simple_filter_with_numeric_operation():
    """Test simple_filter with a numeric operation."""
    def double(value):
        return int(value) * 2
    
    FilterExtension = simple_filter(double)
    env = StrictEnvironment()
    FilterExtension(env)
    
    template = env.from_string('{{ num|double }}')
    result = template.render(num=5)
    assert result == '10'


# LLM-generated content at query #66
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    def uppercase_filter(value):
        """Convert string to uppercase."""
        return value.upper()

    # Create the extension class
    extension_class = simple_filter(uppercase_filter)

    # Verify the extension class has the correct name
    assert extension_class.__name__ == 'uppercase_filter'

    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)

    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension_instance = extension_class(env)

    # Verify the filter was registered in the environment
    assert 'uppercase_filter' in env.filters
    assert env.filters['uppercase_filter'] is uppercase_filter

    # Test that the filter works correctly
    assert env.filters['uppercase_filter']('hello') == 'HELLO'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter extensions can coexist."""
    def lowercase_filter(value):
        """Convert string to lowercase."""
        return value.lower()

    def reverse_filter(value):
        """Reverse a string."""
        return value[::-1]

    # Create extension classes
    lowercase_ext = simple_filter(lowercase_filter)
    reverse_ext = simple_filter(reverse_filter)

    # Verify names are correct
    assert lowercase_ext.__name__ == 'lowercase_filter'
    assert reverse_ext.__name__ == 'reverse_filter'

    # Create environment and register both
    env = StrictEnvironment()
    lowercase_ext(env)
    reverse_ext(env)

    # Verify both filters are registered
    assert 'lowercase_filter' in env.filters
    assert 'reverse_filter' in env.filters

    # Test both filters work
    assert env.filters['lowercase_filter']('HELLO') == 'hello'
    assert env.filters['reverse_filter']('hello') == 'olleh'


def test_simple_filter_with_arguments():
    """Test that simple_filter works with filter functions that take arguments."""
    def replace_filter(value, old, new):
        """Replace old with new in value."""
        return value.replace(old, new)

    extension_class = simple_filter(replace_filter)
    env = StrictEnvironment()
    extension_class(env)

    assert 'replace_filter' in env.filters
    assert env.filters['replace_filter']('hello world', 'world', 'there') == 'hello there'


# LLM-generated content at query #67
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a jinja2 extension."""
    # Define a test filter function
    def test_uppercase(value):
        """Convert value to uppercase."""
        return str(value).upper()
    
    # Decorate the function with simple_filter
    FilterExtension = simple_filter(test_uppercase)
    
    # Verify the extension class is created
    assert FilterExtension is not None
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension name matches the function name
    assert FilterExtension.__name__ == 'test_uppercase'
    
    # Create a StrictEnvironment and register the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    
    # Verify the filter is registered in the environment
    assert 'test_uppercase' in env.filters
    
    # Verify the filter works correctly
    assert env.filters['test_uppercase']('hello') == 'HELLO'
    assert env.filters['test_uppercase']('world') == 'WORLD'


def test_simple_filter_with_multiple_filters():
    """Test that multiple simple_filter extensions can be created."""
    def lowercase_filter(value):
        """Convert value to lowercase."""
        return str(value).lower()
    
    def reverse_filter(value):
        """Reverse the value."""
        return str(value)[::-1]
    
    LowercaseExtension = simple_filter(lowercase_filter)
    ReverseExtension = simple_filter(reverse_filter)
    
    env = StrictEnvironment(extensions=[LowercaseExtension, ReverseExtension])
    
    assert 'lowercase_filter' in env.filters
    assert 'reverse_filter' in env.filters
    assert env.filters['lowercase_filter']('HELLO') == 'hello'
    assert env.filters['reverse_filter']('hello') == 'olleh'


def test_simple_filter_in_template():
    """Test that simple_filter works within actual jinja2 templates."""
    def add_prefix(value):
        """Add prefix to value."""
        return f"PREFIX_{value}"
    
    FilterExtension = simple_filter(add_prefix)
    env = StrictEnvironment(extensions=[FilterExtension])
    
    template = env.from_string("{{ name | add_prefix }}")
    result = template.render(name='test')
    
    assert result == 'PREFIX_test'


# LLM-generated content at query #68
#--------------------------

```python
def test_simple_filter():
    """Test the simple_filter decorator function."""
    # Define a test filter function
    def my_test_filter(value):
        """Test filter that converts to uppercase."""
        return str(value).upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(my_test_filter)
    
    # Verify that the returned class is a subclass of Extension
    assert issubclass(FilterExtension, Extension)
    
    # Verify that the class name matches the filter function name
    assert FilterExtension.__name__ == 'my_test_filter'
    
    # Create a StrictEnvironment and initialize the extension
    env = StrictEnvironment()
    extension_instance = FilterExtension(env)
    
    # Verify that the filter was registered in the environment
    assert 'my_test_filter' in env.filters
    assert env.filters['my_test_filter'] is my_test_filter
    
    # Verify that the filter works correctly
    assert env.filters['my_test_filter']('hello') == 'HELLO'
    assert env.filters['my_test_filter']('test') == 'TEST'


def test_simple_filter_multiple_filters():
    """Test that multiple simple_filter decorators create independent filters."""
    def filter_one(value):
        return str(value) + '_one'
    
    def filter_two(value):
        return str(value) + '_two'
    
    FilterExtensionOne = simple_filter(filter_one)
    FilterExtensionTwo = simple_filter(filter_two)
    
    env = StrictEnvironment()
    FilterExtensionOne(env)
    FilterExtensionTwo(env)
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    assert env.filters['filter_one']('test') == 'test_one'
    assert env.filters['filter_two']('test') == 'test_two'


def test_simple_filter_with_complex_function():
    """Test simple_filter with a more complex filter function."""
    def reverse_string(value):
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    env = StrictEnvironment()
    FilterExtension(env)
    
    assert env.filters['reverse_string']('hello') == 'olleh'
    assert env.filters['reverse_string']('world') == 'dlrow'


# LLM-generated content at query #69
#--------------------------

```python
def test_simple_filter():
    """Test simple_filter decorator creates a proper jinja2 extension."""
    # Define a test filter function
    def uppercase(value):
        """Convert string to uppercase."""
        return str(value).upper()
    
    # Apply the simple_filter decorator
    FilterExtension = simple_filter(uppercase)
    
    # Verify the extension class is created
    assert FilterExtension is not None
    assert issubclass(FilterExtension, Extension)
    
    # Verify the extension class has the correct name
    assert FilterExtension.__name__ == 'uppercase'
    
    # Create a StrictEnvironment and instantiate the extension
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    # Verify the filter is registered in the environment
    assert 'uppercase' in env.filters
    assert env.filters['uppercase'] is uppercase
    
    # Test that the filter works correctly
    result = env.filters['uppercase']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_custom_logic():
    """Test simple_filter with a more complex filter function."""
    def reverse_string(value):
        """Reverse a string."""
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    env = StrictEnvironment()
    extension = FilterExtension(env)
    
    assert 'reverse_string' in env.filters
    result = env.filters['reverse_string']('abc')
    assert result == 'cba'


def test_simple_filter_multiple_extensions():
    """Test that multiple simple_filter extensions can coexist."""
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    PrefixExtension = simple_filter(add_prefix)
    SuffixExtension = simple_filter(add_suffix)
    
    env = StrictEnvironment()
    PrefixExtension(env)
    SuffixExtension(env)
    
    assert 'add_prefix' in env.filters
    assert 'add_suffix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'
    assert env.filters['add_suffix']('test') == 'test_suffix'


