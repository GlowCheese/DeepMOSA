####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / "test_dir"
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles non-existent directory by raising an error
    non_existent_dir = Path(original_dir) / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(FilterExtension)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension has the correct name
    assert FilterExtension.__name__ == "test_filter"


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager handles None dirname correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that the context manager handles non-existent directory
    with pytest.raises(FileNotFoundError):
        with work_in('/non/existent/dir'):
            pass


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / "test_dir"
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in():
    # Setup
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    try:
        # Test entering the context
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)

        # Test exiting the context
        assert os.getcwd() == original_dir

        # Test with None directory
        with work_in(None):
            assert os.getcwd() == original_dir

        # Test nested contexts
        nested_dir = Path(tempfile.mkdtemp())
        with work_in(test_dir):
            with work_in(nested_dir):
                assert os.getcwd() == str(nested_dir)
            assert os.getcwd() == str(test_dir)
        assert os.getcwd() == original_dir

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        if 'nested_dir' in locals():
            shutil.rmtree(nested_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter returns a type that is a subclass of Extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    assert issubclass(test_filter, Extension)

    # Test that the filter is correctly added to the environment
    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in restores directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #15
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Cleanup
    shutil.rmtree(test_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

    finally:
        # Clean up the test directory
        rmtree(test_dir)

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions properly
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    # Should still be back to original directory after exception
    assert os.getcwd() == original_dir


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    # Test that work_in returns to the original directory after exiting
    assert os.getcwd() == original_dir

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    test_dir = Path(tempfile.mkdtemp())
    original_dir = Path.cwd()

    with work_in(test_dir):
        assert Path.cwd() == test_dir

    # Test that work_in returns to the original directory after exiting
    assert Path.cwd() == original_dir

    # Test that work_in works with None (current directory)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles errors and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test error")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    FilterExtension = simple_filter(test_filter)

    # Verify the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(FilterExtension)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

    finally:
        # Clean up
        rmtree(test_dir)

    # Test with None directory (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

        # Test with None directory (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    finally:
        shutil.rmtree(test_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    test_dir = Path(tempfile.mkdtemp())
    original_dir = Path.cwd()

    with work_in(test_dir):
        assert Path.cwd() == test_dir

    # Test that the context manager returns to the original directory after exiting
    assert Path.cwd() == original_dir

    # Test that the context manager handles None directory
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    rmtree(test_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Check that the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a mock environment to test the extension
    env = StrictEnvironment()

    # Initialize the extension
    extension = FilterExtension(env)

    # Check that the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #29
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    TestFilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment(extensions=[TestFilterExtension])

    # Verify the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension class has the correct name
    assert TestFilterExtension.__name__ == "test_filter"


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / "test_work_in_dir"
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == test_filter.__name__

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Create an instance of the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Test the filter works
    test_value = "test"
    result = env.filters[test_filter.__name__](test_value)
    assert result == f"filtered_{test_value}"


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

    finally:
        # Clean up the test directory
        shutil.rmtree(test_dir)

    # Test with None directory (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that it handles exceptions properly
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    # Should still be back to original directory after exception
    assert os.getcwd() == original_dir


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that work_in returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (should stay in current directory)
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"test_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "test_hello"


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    test_dir = Path(tempfile.mkdtemp())
    original_dir = Path.cwd()

    with work_in(test_dir):
        assert Path.cwd() == test_dir

    # Test that work_in returns to the original directory after exiting
    assert Path.cwd() == original_dir

    # Test that work_in works with None (current directory)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that work_in returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles relative paths
    with work_in('..'):
        assert os.getcwd() == os.path.abspath(os.path.join(original_dir, '..'))


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    finally:
        rmtree(test_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that work_in returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles relative paths
    original_dir = os.getcwd()
    with work_in('..'):
        assert os.getcwd() == os.path.dirname(original_dir)
    assert os.getcwd() == original_dir


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that work_in returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #21
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in restores directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"


# LLM-generated content at query #24
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    test_dir = Path(tempfile.mkdtemp())
    original_dir = os.getcwd()

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    # Test that work_in returns to the original directory after exiting
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    finally:
        # Clean up the test directory
        rmtree(test_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in properly handles exceptions and restores the directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir

        # After exiting the context, should be back to original directory
        assert os.getcwd() == original_dir

        # Test with None (should stay in current directory)
        with work_in(None):
            assert os.getcwd() == original_dir

    finally:
        shutil.rmtree(test_dir)


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in restores directory even if an exception occurs
    test_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    os.rmdir(test_dir)


# LLM-generated content at query #32
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == test_filter.__name__

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment(extensions=[extension_class])

    # Verify the filter is registered in the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Test that the filter works in a template
    template = env.from_string("{{ 'test' | filtered_test }}")
    result = template.render()
    assert result == "filtered_test"


# LLM-generated content at query #33
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Apply the extension to the environment
    extension = test_filter
    extension(env)

    # Check that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


