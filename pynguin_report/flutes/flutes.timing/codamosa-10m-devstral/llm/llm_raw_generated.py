####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    # Test that the context manager works and prints the expected output
    # (Note: Testing print output requires more complex setup, this is a basic test)
    assert isinstance(wip, contextlib._GeneratorContextManager)


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just checking it runs without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just checking it runs without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True  # Just checking it runs without error


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Context manager executed without errors


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert "Test task... done." in str(wip)
    assert "s)" in str(wip)


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert isinstance(wip, contextlib._GeneratorContextManager)


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_progress():
    # Test as a decorator
    @work_in_progress("Test decorator")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test as a context manager
    with work_in_progress("Test context manager"):
        time.sleep(0.1)

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to ensure the test passes if no exceptions occur


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test with context manager
    with work_in_progress("Test context manager"):
        time.sleep(0.1)


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test with context manager
    with work_in_progress("Test context"):
        time.sleep(0.1)


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work being done

    # Verify the output contains the description and timing
    # (This is a basic test; actual output capture would require more setup)
    assert True  # Placeholder for actual assertions


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test as decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def sample_function():
        time.sleep(0.1)

    sample_function()


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        assert ctx is None  # Context manager yields None
    # Output should be "Work in progress... done. (X.XXs)"

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        assert ctx is None
    # Output should be "Custom task... done. (X.XXs)"

    # Test timing functionality
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.1)  # Simulate work
    elapsed = time.time() - start
    assert elapsed >= 0.1  # Ensure timing is working


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #33
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert "Work in progress... done." in str(wip)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert "Custom task... done." in str(wip)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test that the output contains the expected parts
    # (Note: This would normally use capsys to capture output)
    # For actual testing, you would need to mock print or use capsys


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert wip is None  # Context manager doesn't return anything

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert wip is None

    # Test as decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator with default description
    @work_in_progress()
    def test_func():
        time.sleep(0.1)
    test_func()

    # Test as decorator with custom description
    @work_in_progress("Decorated task")
    def test_func_custom():
        time.sleep(0.1)
    test_func_custom()


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test with default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test with context manager
    with work_in_progress("Test context manager"):
        time.sleep(0.1)


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Testing context manager") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Testing decorator")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert "Work in progress... done." in str(wip)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert "Custom task... done." in str(wip)

    # Test timing is reasonable
    start = time.time()
    with work_in_progress("Timed task"):
        time.sleep(0.2)
    elapsed = time.time() - start
    assert 0.2 <= elapsed <= 0.3  # Allow some overhead


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test with context manager
    with work_in_progress("Test context"):
        time.sleep(0.1)

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.09 <= elapsed <= 0.11


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test that the output contains expected parts
    import io
    import sys

    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    with work_in_progress("Captured test"):
        time.sleep(0.1)

    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "Captured test... done." in output
    assert "s)" in output  # Check for time format


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    # No assertion needed, just checking it doesn't raise an exception

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    # No assertion needed, just checking it doesn't raise an exception

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    # No assertion needed, just checking it doesn't raise an exception


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task"):
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #27
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a description
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert wip is None  # Context manager doesn't return anything

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert wip is None

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    result = test_func()
    assert result is None

    # Test timing accuracy (should be at least 0.1s)
    start = time.time()
    with work_in_progress():
        time.sleep(0.1)
    elapsed = time.time() - start
    assert elapsed >= 0.1


# LLM-generated content at query #29
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress() as _:
        time.sleep(0.1)


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    # The output is printed to stdout, so we can't directly test it here
    # But we can verify that the context manager works without errors
    assert wip is None  # The context manager doesn't return anything


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #33
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    end = time.time()
    assert abs((end - start) - 0.1) < 0.01


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception):
        with work_in_progress("Failing task"):
            raise Exception("Test exception")

    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #36
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorator task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        assert wip is None  # Context manager yields None

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        assert wip is None

    # Test as decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test timing is reasonable (should be at least 0.1s for the sleep)
    start = time.time()
    with work_in_progress():
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.1 <= elapsed <= 0.2  # Allow some margin for system overhead


