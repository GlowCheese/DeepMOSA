####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    with work_in("/tmp"):
        assert os.getcwd() == "/tmp"

    # Test that work_in returns to the original directory after exiting
    original_dir = os.getcwd()
    with work_in("/tmp"):
        pass
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory (should stay in current directory)
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter returns a subclass of Extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    assert issubclass(test_filter, Extension)

    # Test that the filter is added to the environment
    env = StrictEnvironment()
    ext = test_filter(env)
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Test that the filter name is preserved
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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

    # Test that the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #7
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
    shutil.rmtree(test_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension is properly named
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #9
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension using the decorator
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(extension_class)

    # Verify the filter is registered in the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #10
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter is registered in the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


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
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory after exit
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager handles None dirname correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that the context manager handles exceptions and still returns to the original directory
    with pytest.raises(ValueError):
        with work_in('/tmp'):
            raise ValueError
    assert os.getcwd() == original_dir


# LLM-generated content at query #13
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
        rmtree(test_dir)

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir


# LLM-generated content at query #14
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


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

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    def test_filter(value):
        return f"test_{value}"

    # Create the extension class
    TestFilterExtension = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert TestFilterExtension.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(TestFilterExtension)

    # Verify the filter is available in the environment
    assert "test_filter" in env.filters

    # Test the filter works as expected
    template = env.from_string("{{ 'value' | test_filter }}")
    assert template.render() == "test_value"


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager works with None
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that the context manager works with Path objects
    with work_in(Path('/tmp')):
        assert os.getcwd() == '/tmp'


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    # Test that work_in returns to the original directory after exiting
    assert os.getcwd() == original_dir

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
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

    # Test that work_in handles None directory (should stay in current directory)
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #21
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

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles non-existent directory
    non_existent_dir = Path(original_dir) / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #22
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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    assert os.getcwd() == original_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    shutil.rmtree(test_dir)


# LLM-generated content at query #26
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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #27
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

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Cleanup
    test_dir.rmdir()


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

    # Test that work_in handles None directory correctly
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


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #30
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


# LLM-generated content at query #31
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


# LLM-generated content at query #32
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


# LLM-generated content at query #33
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
    if test_dir.exists():
        test_dir.rmdir()


# LLM-generated content at query #34
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


# LLM-generated content at query #35
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

    # Test that work_in handles errors and still returns to the original directory
    with pytest.raises(Exception):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise Exception("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #36
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

    # Test that work_in handles errors and still returns to the original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test error")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #37
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


# LLM-generated content at query #38
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


# LLM-generated content at query #39
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager works with None
    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #40
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory after exit
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager handles None dirname correctly
    with work_in(None):
        assert os.getcwd() == original_dir


# LLM-generated content at query #43
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

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #44
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment to test the extension
    from jinja2 import Environment
    env = Environment(extensions=[test_filter])

    # Verify the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension class is properly named
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #45
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #46
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    FilterExtension = simple_filter(test_filter)

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment(extensions=[FilterExtension])
    template = env.from_string("{{ 'test' | test_filter }}")

    # Render the template and check the result
    assert template.render() == "filtered_test"


# LLM-generated content at query #47
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter wraps a function in a jinja2 extension
    def test_filter(value):
        return value.upper()

    # Create the extension
    extension_class = simple_filter(test_filter)

    # Verify the extension is properly named
    assert extension_class.__name__ == 'test_filter'

    # Create a mock environment
    env = StrictEnvironment()

    # Initialize the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test the filter works
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #48
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test with a different filter
    @simple_filter
    def another_filter(value):
        return value.upper()

    env.add_extension(another_filter)
    assert 'another_filter' in env.filters
    assert env.filters['another_filter']("test") == "TEST"


# LLM-generated content at query #49
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension
    extension_class = simple_filter(test_filter)

    # Verify the extension class is properly named
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    env = StrictEnvironment()

    # Instantiate the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #50
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


# LLM-generated content at query #51
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #52
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #53
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Create an instance of the extension
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Verify the extension is properly named
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #54
#--------------------------

```python
def test_simple_filter():
    # Test that the decorated function is wrapped in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"

    # Check that the extension is an instance of Extension
    assert isinstance(extension, Extension)


# LLM-generated content at query #55
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'

    # Verify the extension class name matches the filter function name
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #56
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    def test_filter(value):
        return value.upper()

    # Apply the decorator
    TestFilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = TestFilterExtension(env)

    # Check that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Check that the extension name matches the filter function name
    assert TestFilterExtension.__name__ == test_filter.__name__


# LLM-generated content at query #57
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Check that the extension class name matches the filter function name
    assert extension.__name__ == 'test_filter'


# LLM-generated content at query #58
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter is registered in the environment
    assert "test_filter" in env.filters

    # Test the filter function
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #59
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Verify the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #60
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


# LLM-generated content at query #61
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is properly registered
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #62
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered
    assert test_filter.__name__ in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Test that the extension is properly named
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #65
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


# LLM-generated content at query #66
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator works correctly
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    extension = test_filter
    extension(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #67
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
    from jinja2 import Environment
    env = Environment(extensions=[FilterExtension])

    # Verify the filter is added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Verify the extension is properly named
    assert extension.__name__ == 'test_filter'


# LLM-generated content at query #70
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


# LLM-generated content at query #71
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Apply the decorator
    TestFilterExtension = simple_filter(test_filter)

    # Create a jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(TestFilterExtension)

    # Check that the filter is available in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test that the filter works as expected
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #72
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter returns a valid Jinja2 Extension class
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment to test the extension
    env = StrictEnvironment()

    # Apply the extension to the environment
    extension_class = test_filter
    extension_instance = extension_class(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Verify the extension class name matches the filter function name
    assert extension_class.__name__ == 'test_filter'


# LLM-generated content at query #73
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Create an instance of the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works as expected
    test_value = "example"
    assert env.filters["test_filter"](test_value) == "filtered_example"


# LLM-generated content at query #74
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


# LLM-generated content at query #75
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


# LLM-generated content at query #76
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    ext = test_filter(env)

    # Check that the filter is added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #77
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #78
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

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #79
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Check that the extension is properly named
    assert extension.__name__ == 'test_filter'


# LLM-generated content at query #80
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Verify the extension is properly named
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #81
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters

    # Test the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #82
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


# LLM-generated content at query #83
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a test environment
    env = StrictEnvironment()

    # Add the decorated filter to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"

    # Test that the filter name matches the function name
    assert 'test_filter' in env.filters


# LLM-generated content at query #84
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter is registered
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #85
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is properly registered
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #86
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #87
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

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Test the filter works
    test_value = "test"
    assert env.filters[test_filter.__name__](test_value) == f"filtered_{test_value}"


# LLM-generated content at query #88
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test with a different filter
    @simple_filter
    def another_filter(value):
        return value.upper()

    env.add_extension(another_filter)
    assert 'another_filter' in env.filters
    assert env.filters['another_filter']("test") == "TEST"


# LLM-generated content at query #89
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


# LLM-generated content at query #90
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    extension_class = simple_filter(test_filter)

    # Check that the extension class is created correctly
    assert extension_class.__name__ == test_filter.__name__

    # Check that the extension can be instantiated and adds the filter
    env = StrictEnvironment()
    extension = extension_class(env)
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Check that the filter works correctly
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #91
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"


# LLM-generated content at query #92
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Verify the extension class properties
    assert extension.__name__ == 'test_filter'


# LLM-generated content at query #93
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'hello' | test_filter }}")
    rendered = template.render()
    assert rendered == "HELLO"

    # Test that the filter is accessible by the function name
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('world') == 'WORLD'


# LLM-generated content at query #94
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Check that the extension class has the correct name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #95
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


# LLM-generated content at query #96
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter returns a valid Extension class
    def dummy_filter(value):
        return value.upper()

    extension_class = simple_filter(dummy_filter)

    # Check that the returned object is a class
    assert isinstance(extension_class, type)

    # Check that the class is a subclass of Extension
    assert issubclass(extension_class, Extension)

    # Check that the class name matches the filter function name
    assert extension_class.__name__ == dummy_filter.__name__

    # Test that the extension properly registers the filter
    env = StrictEnvironment()
    extension_instance = extension_class(env)

    # Check that the filter was added to the environment
    assert dummy_filter.__name__ in env.filters
    assert env.filters[dummy_filter.__name__] is dummy_filter

    # Test that the filter works in a template
    template = env.from_string("{{ 'hello' | dummy_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #97
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Verify the extension is properly named
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #98
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == 'HELLO'


# LLM-generated content at query #99
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


# LLM-generated content at query #100
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Verify the extension class name matches the filter name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #101
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


# LLM-generated content at query #102
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = "{{ 'test' | test_filter }}"
    result = env.from_string(template).render()
    assert result == "filtered_test"


# LLM-generated content at query #103
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter wraps a function in a Jinja2 extension."""
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created with the correct name
    assert extension_class.__name__ == 'test_filter'

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Instantiate the extension
    extension = extension_class(env)

    # Verify the filter is added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test the filter works
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #104
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"

    # Verify the extension class name matches the filter function name
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #105
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


# LLM-generated content at query #106
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the extension is properly set up
    assert isinstance(extension, Extension)
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Test that the filter works correctly
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #107
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters

    # Test the filter works correctly
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #108
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #109
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    from jinja2 import Environment
    env = Environment(extensions=[test_filter])

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Test that the extension has the correct name
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #110
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a Jinja2 extension with the correct filter
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"

    # Test that the extension is named after the filter function
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #111
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test with a different filter
    @simple_filter
    def upper_filter(value):
        return value.upper()

    env.add_extension(upper_filter)
    assert 'upper_filter' in env.filters
    assert env.filters['upper_filter']("hello") == "HELLO"


# LLM-generated content at query #112
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test with a different filter
    @simple_filter
    def another_filter(value):
        return value.upper()

    env.add_extension(another_filter)
    assert 'another_filter' in env.filters
    assert env.filters['another_filter']("test") == "TEST"


# LLM-generated content at query #113
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


# LLM-generated content at query #114
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify it's a subclass of Extension
    assert issubclass(extension_class, Extension)

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(extension_class)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test that the filter works
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #115
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    def test_filter(input_str):
        return input_str.upper()

    # Apply the decorator
    TestFilterExtension = simple_filter(test_filter)

    # Verify the extension is created correctly
    assert TestFilterExtension.__name__ == 'test_filter'

    # Create a mock environment
    env = StrictEnvironment()

    # Initialize the extension
    extension = TestFilterExtension(env)

    # Verify the filter is added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test the filter works as expected
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #116
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"

    # Test that the filter name matches the function name
    assert 'test_filter' in env.filters

    # Test that the extension class name matches the function name
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #117
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


# LLM-generated content at query #118
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


# LLM-generated content at query #119
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    env = StrictEnvironment()

    # Instantiate the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works as expected
    assert env.filters["test_filter"]("test_value") == "filtered_test_value"


# LLM-generated content at query #120
#--------------------------

```python
def test_simple_filter():
    # Test that the filter is correctly added to the environment
    def test_filter(value):
        return f"filtered_{value}"

    extension_class = simple_filter(test_filter)
    env = StrictEnvironment()
    extension = extension_class(env)

    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"

    # Test that the extension class has the correct name
    assert extension_class.__name__ == test_filter.__name__


# LLM-generated content at query #121
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

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert test_filter.__name__ in env.filters

    # Test that the filter works as expected
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #122
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works as expected
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"


# LLM-generated content at query #123
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is available in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #124
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ value | test_filter }}")
    result = template.render(value="test")
    assert result == "filtered_test"


# LLM-generated content at query #125
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment(extensions=[test_filter])

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #126
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


# LLM-generated content at query #127
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


# LLM-generated content at query #128
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)
    extension = FilterExtension(StrictEnvironment())

    # Check that the filter was added to the environment
    assert 'test_filter' in extension.environment.filters
    assert extension.environment.filters['test_filter'](123) == "filtered_123"

    # Check that the extension name matches the filter function name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #129
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(extension_class)

    # Verify the filter is registered
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #130
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #131
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


# LLM-generated content at query #132
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


# LLM-generated content at query #133
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator properly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Create the extension
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Verify the extension class name matches the filter function name
    assert extension.__name__ == test_filter.__name__


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"
    assert issubclass(extension_class, Extension)

    # Test that the extension works in a Jinja2 environment
    env = StrictEnvironment()
    extension = extension_class(env)

    # Verify the filter is added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works as expected
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #3
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
        shutil.rmtree(test_dir)

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in works with string path
    test_dir_str = str(test_dir)
    try:
        with work_in(test_dir_str):
            assert os.getcwd() == test_dir_str
    finally:
        shutil.rmtree(test_dir_str)


# LLM-generated content at query #4
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


# LLM-generated content at query #5
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

    # Test that work_in handles None directory (should stay in current dir)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    rmtree(test_dir)


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    rmtree(test_dir)


# LLM-generated content at query #8
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
    shutil.rmtree(test_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and back
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


# LLM-generated content at query #10
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

    # Test that work_in handles None directory (should stay in current dir)
    with work_in(None):
        assert Path.cwd() == original_dir

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

    # Test that work_in handles None directory correctly
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


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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
        shutil.rmtree(test_dir)

    # Test with None directory (should not change directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test nested context managers
    nested_dir = Path(tempfile.mkdtemp())
    try:
        with work_in(test_dir):
            inner_original = os.getcwd()
            with work_in(nested_dir):
                assert os.getcwd() == str(nested_dir)
            assert os.getcwd() == inner_original
    finally:
        shutil.rmtree(nested_dir)


# LLM-generated content at query #17
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

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()
    try:
        with work_in(test_dir):
            assert os.getcwd() == test_dir
        # After exiting the context, should return to original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)

    # Test that work_in works with None (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions properly
    test_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise ValueError("Test exception")
        # After exception, should still return to original directory
        assert os.getcwd() == original_dir
    finally:
        rmtree(test_dir)


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

    # Test that work_in handles None directory (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in properly cleans up even if an exception occurs
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

    # Test that work_in works with string path
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)

    # Cleanup
    shutil.rmtree(test_dir)


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / 'test_dir'
    test_dir.mkdir(exist_ok=True)

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
    test_dir.rmdir()


# LLM-generated content at query #23
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

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    test_dir = '/tmp/test_work_in'
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    # Test that the context manager returns to the original directory after exit
    original_dir = os.getcwd()
    with work_in(test_dir):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in():
    # Test that work_in changes directory and returns to original
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    # Test that work_in works with None
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


# LLM-generated content at query #27
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"

    # Test that the filter name matches the function name
    assert 'test_filter' in env.filters


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    with work_in():
        assert os.getcwd() == original_dir

    assert os.getcwd() == original_dir

    shutil.rmtree(test_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
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


# LLM-generated content at query #30
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


# LLM-generated content at query #31
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

    # Test that work_in handles None directory (stays in current directory)
    original_dir = os.getcwd()
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles relative paths
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'subdir')
        os.makedirs(subdir)
        with work_in(subdir):
            assert os.getcwd() == subdir


# LLM-generated content at query #32
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

    # Test that work_in handles Path objects
    with work_in(Path('/tmp')):
        assert os.getcwd() == '/tmp'


# LLM-generated content at query #33
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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


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

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #35
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
        shutil.rmtree(test_dir)

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a Jinja2 environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "HELLO"


# LLM-generated content at query #3
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

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #4
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

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in():
    # Setup
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    # Test
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    # Verify
    assert os.getcwd() == original_dir

    # Cleanup
    shutil.rmtree(test_dir)


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir)

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir
    assert os.getcwd() == original_dir

    # Test that work_in restores directory even if an exception occurs
    test_dir = Path(tempfile.mkdtemp())
    try:
        with pytest.raises(ValueError):
            with work_in(test_dir):
                assert os.getcwd() == str(test_dir)
                raise ValueError("Test exception")
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir)


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #9
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

    # Test that work_in handles non-existent directory
    non_existent_dir = Path(original_dir) / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass

    # Clean up
    test_dir.rmdir()


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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


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

    # Test that work_in handles None directory correctly
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


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
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
            assert os.getcwd() == str(test_dir)
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

    # Test that work_in works with None (should stay in current directory)
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


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
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


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    test_dir = Path(tempfile.mkdtemp())
    original_dir = os.getcwd()

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    # Test that the context manager returns to the original directory
    assert os.getcwd() == original_dir

    # Test that the context manager handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #21
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

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #22
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
        rmtree(test_dir)

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir


# LLM-generated content at query #23
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

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    rmtree(test_dir)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #26
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


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in():
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / 'test_work_in_dir'
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    test_dir.rmdir()


# LLM-generated content at query #29
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function
    # in a simplified jinja2 extension

    # Define a simple filter function
    def test_filter(value):
        return value.upper()

    # Apply the decorator
    TestFilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Instantiate the extension
    extension = TestFilterExtension(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Check that the extension class has the correct name
    assert TestFilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #30
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
    test_dir = 'test_dir'
    os.makedirs(test_dir, exist_ok=True)
    with work_in(test_dir):
        assert os.getcwd().endswith(test_dir)
    assert os.getcwd() == original_dir
    shutil.rmtree(test_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
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


# LLM-generated content at query #33
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

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    rmtree(test_dir)


# LLM-generated content at query #34
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
        shutil.rmtree(test_dir)

    # Test that work_in works with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions properly
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    # Should still return to original directory after exception
    assert os.getcwd() == original_dir


# LLM-generated content at query #35
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

    # Test with None directory (should stay in current directory)
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #36
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is properly registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension class has the correct name
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #37
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

    # Test that work_in returns to original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #38
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter function creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Check that the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a test environment and add the extension
    env = StrictEnvironment()
    env.add_extension(FilterExtension)

    # Verify that the filter is added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #39
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'

    # Verify the extension class name matches the filter function name
    assert extension.__name__ == 'test_filter'


# LLM-generated content at query #40
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


# LLM-generated content at query #41
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

    # Test that work_in handles errors and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test error")

    assert os.getcwd() == original_dir


# LLM-generated content at query #42
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
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #43
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'test' | test_filter }}")
    result = template.render()
    assert result == "filtered_test"

    # Test that the filter name matches the function name
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #44
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return value.upper()

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == 'test_filter'

    # Create an instance of the extension
    env = StrictEnvironment()
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test the filter works as expected
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #45
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test that the extension class has the correct name
    assert test_filter.__name__ == 'test_filter'


# LLM-generated content at query #46
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the extension is properly set up
    assert isinstance(extension, Extension)
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Test the filter function works
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #47
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
        shutil.rmtree(test_dir)

    # Test that work_in stays in the current directory if None is passed
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in restores the original directory even if an exception occurs
    test_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(Exception):
            with work_in(test_dir):
                assert os.getcwd() == test_dir
                raise Exception("Test exception")
        assert os.getcwd() == original_dir
    finally:
        shutil.rmtree(test_dir)


# LLM-generated content at query #48
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


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in():
    original_dir = os.getcwd()
    test_dir = "/tmp/test_work_in"
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir

    os.rmdir(test_dir)


# LLM-generated content at query #50
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory
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


# LLM-generated content at query #51
#--------------------------

```python
def test_work_in():
    # Test that the context manager changes to the specified directory
    with work_in('/tmp'):
        assert os.getcwd() == '/tmp'

    # Test that the context manager returns to the original directory after exit
    original_dir = os.getcwd()
    with work_in('/tmp'):
        pass
    assert os.getcwd() == original_dir

    # Test that the context manager handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that the context manager handles Path object
    with work_in(Path('/tmp')):
        assert os.getcwd() == '/tmp'


# LLM-generated content at query #52
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


# LLM-generated content at query #53
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


# LLM-generated content at query #54
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


# LLM-generated content at query #55
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == test_dir
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #56
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


# LLM-generated content at query #57
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

    # Test that work_in handles non-existent directory
    non_existent_dir = Path(original_dir) / "non_existent"
    with pytest.raises(FileNotFoundError):
        with work_in(non_existent_dir):
            pass


# LLM-generated content at query #58
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


# LLM-generated content at query #59
#--------------------------

```python
def test_work_in():
    # Test that the working directory changes and is restored
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that it handles exceptions properly
    with pytest.raises(ValueError):
        with work_in(test_dir):
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #60
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

    # Test that work_in handles exceptions and still returns to original directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #61
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / 'test_dir'
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #62
#--------------------------

```python
def test_work_in():
    # Test that work_in changes the current working directory
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    # Test that work_in restores the original directory after exiting
    assert os.getcwd() == original_dir

    # Test that work_in handles None as dirname
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #63
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / "test_dir"
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    # After exiting the context, should be back to original directory
    assert os.getcwd() == original_dir

    # Test with None (should stay in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #64
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    # Test that work_in returns to the original directory after exiting
    assert os.getcwd() == original_dir

    # Test that work_in handles None directory
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #65
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and back
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in works with None (current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in handles exceptions and still restores directory
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #66
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


# LLM-generated content at query #67
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

    # Test that work_in handles None directory
    with work_in(None):
        assert Path.cwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter wraps a function in a jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    filter_extension = simple_filter(test_filter)

    # Check that the returned object is a subclass of Extension
    assert issubclass(filter_extension, Extension)

    # Create a mock environment to test the extension
    env = StrictEnvironment()

    # Initialize the extension with the environment
    extension_instance = filter_extension(env)

    # Verify that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] == test_filter

    # Test that the filter works as expected
    assert env.filters[test_filter.__name__]("test_value") == "filtered_test_value"


# LLM-generated content at query #70
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is properly created
    assert extension_class.__name__ == "test_filter"
    assert issubclass(extension_class, Extension)

    # Create a mock environment
    env = StrictEnvironment()

    # Create an instance of the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works as expected
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #71
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


# LLM-generated content at query #72
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


# LLM-generated content at query #73
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Verify the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment(extensions=[FilterExtension])

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #74
#--------------------------

```python
def test_work_in():
    # Test that work_in changes to the specified directory and returns to the original
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir

    # Test that work_in handles None directory correctly
    with work_in(None):
        assert os.getcwd() == original_dir

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #75
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

    # Add the extension to the environment
    extension = test_filter
    extension(env)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #76
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

    # Test that work_in returns to the original directory even if an exception occurs
    with pytest.raises(ValueError):
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")

    assert os.getcwd() == original_dir

    # Clean up
    test_dir.rmdir()


# LLM-generated content at query #77
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


# LLM-generated content at query #78
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


# LLM-generated content at query #79
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


# LLM-generated content at query #80
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

    # Test that work_in handles None directory (stays in current directory)
    with work_in(None):
        assert os.getcwd() == original_dir

    # Clean up
    shutil.rmtree(test_dir)


# LLM-generated content at query #81
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment to test the extension
    env = StrictEnvironment()
    extension = test_filter(env)

    # Verify the extension is properly registered
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__]("test") == "filtered_test"


# LLM-generated content at query #82
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


# LLM-generated content at query #83
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('hello') == 'HELLO'


# LLM-generated content at query #84
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = "{{ 'test' | test_filter }}"
    rendered = env.from_string(template).render()
    assert rendered == "filtered_test"


# LLM-generated content at query #85
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
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


# LLM-generated content at query #86
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


# LLM-generated content at query #87
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
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


# LLM-generated content at query #88
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Verify the extension class name matches the filter name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #89
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


# LLM-generated content at query #90
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


# LLM-generated content at query #91
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is registered and works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #92
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Check that the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(FilterExtension)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"

    # Test with a different filter function
    def another_filter(value):
        return value.upper()

    AnotherFilterExtension = simple_filter(another_filter)
    assert AnotherFilterExtension.__name__ == "another_filter"

    env.add_extension(AnotherFilterExtension)
    assert "another_filter" in env.filters
    assert env.filters["another_filter"]("test") == "TEST"


# LLM-generated content at query #93
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class has the correct name
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Initialize the extension
    extension = extension_class(env)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test the filter works as expected
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #94
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = test_filter(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Verify the extension is properly named
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #95
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a valid Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension
    extension_class = simple_filter(test_filter)

    # Verify the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(extension_class)

    # Verify the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #96
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


# LLM-generated content at query #97
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #98
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
    extension = test_filter(env)

    # Verify that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #99
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


# LLM-generated content at query #100
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"

    # Test with a different filter
    @simple_filter
    def another_filter(value):
        return value.upper()

    env.add_extension(another_filter)
    assert 'another_filter' in env.filters
    assert env.filters['another_filter']("hello") == "HELLO"


# LLM-generated content at query #101
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Check that the extension is properly named
    assert FilterExtension.__name__ == "test_filter"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(FilterExtension)

    # Check that the filter is registered in the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #102
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #103
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

    # Verify that the filter is registered
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #104
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is registered
    assert 'test_filter' in env.filters

    # Test the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


# LLM-generated content at query #105
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter returns a valid Extension type
    @simple_filter
    def test_filter(value):
        return value.upper()

    # Check that the returned object is a subclass of Extension
    assert issubclass(test_filter, Extension)

    # Check that the extension name matches the filter function name
    assert test_filter.__name__ == 'test_filter'

    # Test that the filter is properly added to the environment
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Verify the filter is in the environment's filters
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

    # Test that the filter works as expected
    template = env.from_string('{{ value | test_filter }}')
    assert template.render(value='hello') == 'HELLO'


# LLM-generated content at query #106
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #107
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()
    extension = FilterExtension(env)

    # Verify the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'

    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == 'test_filter'


# LLM-generated content at query #108
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is registered and works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"

    # Test that the extension class has the correct name
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #109
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Check that the extension class is created correctly
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    from jinja2 import Environment
    env = Environment()

    # Create an instance of the extension
    extension = extension_class(env)

    # Verify that the filter was added to the environment
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

    # Test that the filter works as expected
    assert env.filters["test_filter"]("test") == "filtered_test"


# LLM-generated content at query #110
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    FilterExtension = simple_filter(test_filter)

    # Check that the extension is properly named
    assert FilterExtension.__name__ == 'test_filter'

    # Create a mock environment to test the extension
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(FilterExtension)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']('test') == 'filtered_test'


# LLM-generated content at query #111
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


# LLM-generated content at query #112
#--------------------------

```python
def test_simple_filter():
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    env = StrictEnvironment()
    env.add_extension(test_filter)

    template = "{{ 'test' | test_filter }}"
    result = env.from_string(template).render()
    assert result == "filtered_test"


# LLM-generated content at query #113
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


# LLM-generated content at query #114
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered in the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #115
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment and add the extension
    env = StrictEnvironment()
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"

    # Test that the filter name matches the function name
    assert 'test_filter' in env.filters


# LLM-generated content at query #116
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Create an instance of the extension
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Check that the extension is properly named
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #117
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter correctly wraps a function in a Jinja2 extension
    def test_filter(value):
        return f"filtered_{value}"

    # Create the extension class
    extension_class = simple_filter(test_filter)

    # Verify the extension class is properly named
    assert extension_class.__name__ == "test_filter"

    # Create a mock environment
    mock_env = type('MockEnvironment', (), {
        'filters': {}
    })()

    # Instantiate the extension
    extension = extension_class(mock_env)

    # Verify the filter was added to the environment
    assert 'test_filter' in mock_env.filters
    assert mock_env.filters['test_filter'] is test_filter

    # Test the filter function works as expected
    test_value = "test"
    assert mock_env.filters['test_filter'](test_value) == "filtered_test"


# LLM-generated content at query #118
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


# LLM-generated content at query #119
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a valid Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #120
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator correctly wraps a function
    # and makes it available as a Jinja2 filter

    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a Jinja2 environment
    env = StrictEnvironment()

    # Add the filter extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is available in the environment
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"

    # Test that the filter works with different input types
    template = env.from_string("{{ 123 | test_filter }}")
    assert template.render() == "filtered_123"

    # Test that the filter's name is correctly set
    assert test_filter.__name__ == "test_filter"


# LLM-generated content at query #121
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


# LLM-generated content at query #122
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator wraps a function in a jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Create an instance of the extension
    extension = test_filter(env)

    # Check that the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Check that the extension is named after the filter function
    assert extension.__name__ == test_filter.__name__


# LLM-generated content at query #123
#--------------------------

```python
def test_simple_filter():
    """Test that simple_filter correctly wraps a function in a Jinja2 extension."""

    # Define a simple filter function
    def test_filter(value):
        return value.upper()

    # Create the extension class using simple_filter
    TestFilterExtension = simple_filter(test_filter)

    # Create a mock environment
    class MockEnvironment:
        def __init__(self):
            self.filters = {}

    env = MockEnvironment()

    # Instantiate the extension
    extension = TestFilterExtension(env)

    # Verify the filter was added to the environment
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter

    # Verify the extension class name matches the filter function name
    assert TestFilterExtension.__name__ == test_filter.__name__


# LLM-generated content at query #124
#--------------------------

```python
def test_simple_filter():
    # Test that the simple_filter decorator creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a mock environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Check that the filter was added to the environment
    assert 'test_filter' in env.filters
    assert env.filters['test_filter']("test") == "filtered_test"


# LLM-generated content at query #125
#--------------------------

```python
def test_simple_filter():
    # Test that simple_filter creates a proper Jinja2 extension
    @simple_filter
    def test_filter(value):
        return f"filtered_{value}"

    # Create a test environment
    env = StrictEnvironment()

    # Add the extension to the environment
    env.add_extension(test_filter)

    # Test that the filter is registered
    assert 'test_filter' in env.filters

    # Test that the filter works correctly
    template = env.from_string("{{ 'test' | test_filter }}")
    assert template.render() == "filtered_test"


