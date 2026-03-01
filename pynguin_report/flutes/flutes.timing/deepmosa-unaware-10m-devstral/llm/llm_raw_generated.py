####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #3
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert isinstance(wip, contextlib._GeneratorContextManager)


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test with context manager
    with work_in_progress("Test context manager"):
        time.sleep(0.1)

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #6
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None


# LLM-generated content at query #8
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
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress():
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15  # Allow some margin for system overhead


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #10
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


# LLM-generated content at query #11
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

    # Test that the output contains expected parts
    import io
    import sys

    output = io.StringIO()
    sys.stdout = output

    with work_in_progress("Custom task"):
        time.sleep(0.1)

    sys.stdout = sys.__stdout__
    output_str = output.getvalue()

    assert "Custom task... done." in output_str
    assert "s)" in output_str  # Check for time measurement


# LLM-generated content at query #12
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)  # Simulate work

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


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
    # Test the context manager functionality
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator functionality
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(AttributeError):
        with work_in_progress() as ctx:
            _ = ctx.nonexistent_attribute

    @work_in_progress("Decorated function")
    def dummy_func():
        time.sleep(0.1)

    dummy_func()


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test with no description
    with work_in_progress() as wip:
        time.sleep(0.1)


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as _:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as _:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
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


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.1)

    test_function()


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

    # Test that the output contains expected parts
    import io
    import sys
    from unittest.mock import patch

    output = io.StringIO()
    with patch('sys.stdout', new=output):
        with work_in_progress("Custom task"):
            time.sleep(0.1)

    output_str = output.getvalue()
    assert "Custom task... done." in output_str
    assert "s)" in output_str  # Check for time measurement


# LLM-generated content at query #31
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
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #32
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
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Context manager completed successfully

def test_work_in_progress_decorator():
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True  # Decorator completed successfully


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    end = time.time()
    assert abs((end - start) - 0.1) < 0.05


# LLM-generated content at query #39
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #40
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test with decorator
    @work_in_progress("Test decorator")
    def dummy_function():
        time.sleep(0.1)  # Simulate some work

    dummy_function()


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test timing accuracy (should be at least 0.1 seconds)
    start = time.time()
    with work_in_progress() as wip:
        time.sleep(0.1)
    elapsed = time.time() - start
    assert elapsed >= 0.1


# LLM-generated content at query #43
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate some work

    dummy_function()


# LLM-generated content at query #44
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #45
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


# LLM-generated content at query #46
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #47
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #48
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #50
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


# LLM-generated content at query #51
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to ensure the test passes if no exceptions occur


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test operation") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #53
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate some work

    test_func()


# LLM-generated content at query #54
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


# LLM-generated content at query #55
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #56
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality
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


# LLM-generated content at query #57
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()


# LLM-generated content at query #58
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


# LLM-generated content at query #59
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator functionality
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #60
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    # No assertion needed, just checking it runs without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    # No assertion needed, just checking it runs without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    # No assertion needed, just checking it runs without error


# LLM-generated content at query #61
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


# LLM-generated content at query #62
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


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #65
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


# LLM-generated content at query #66
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #67
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to ensure the context manager completes


# LLM-generated content at query #68
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #69
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #70
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    assert isinstance(wip, contextlib._GeneratorContextManager)

    with work_in_progress() as wip:
        time.sleep(0.1)

    assert isinstance(wip, contextlib._GeneratorContextManager)


# LLM-generated content at query #71
#--------------------------

```python
def test_work_in_progress():
    # Test as a context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test as a decorator
    @work_in_progress("Decorated test")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #72
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just to check if the context manager completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just to check if the context manager completes without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True  # Just to check if the decorator completes without error


# LLM-generated content at query #73
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
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #74
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #75
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #76
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #77
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality
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


# LLM-generated content at query #78
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


# LLM-generated content at query #79
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #80
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #81
#--------------------------

```python
def test_work_in_progress():
    # Test basic context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate some work


# LLM-generated content at query #82
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()


# LLM-generated content at query #83
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

    # Test with default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #84
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #85
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #86
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


# LLM-generated content at query #87
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        assert ctx is None  # Context manager yields None

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        assert ctx is None
        time.sleep(0.1)  # Simulate some work

    # Test as decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test timing is reasonable (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15  # Allow some margin for system overhead


# LLM-generated content at query #88
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


# LLM-generated content at query #89
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #90
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #91
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #92
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate some work

    test_func()


# LLM-generated content at query #93
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


# LLM-generated content at query #94
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #95
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


# LLM-generated content at query #96
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #97
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    # The context manager should have printed the description and timing
    # We can't easily capture stdout in pytest without additional fixtures,
    # so we'll just verify the context manager works without errors
    assert wip is None  # The context manager yields nothing

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()  # Should print timing information without errors


# LLM-generated content at query #98
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress() as _:
        time.sleep(0.1)


# LLM-generated content at query #99
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator functionality
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #100
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #101
#--------------------------

```python
def test_work_in_progress(capsys):
    # Test with decorator
    @work_in_progress("Test decorator")
    def dummy_func():
        time.sleep(0.1)

    dummy_func()
    captured = capsys.readouterr()
    assert "Test decorator... done. (0.10s)" in captured.out

    # Test with context manager
    with work_in_progress("Test context"):
        time.sleep(0.1)

    captured = capsys.readouterr()
    assert "Test context... done. (0.10s)" in captured.out

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    captured = capsys.readouterr()
    assert "Work in progress... done. (0.10s)" in captured.out


# LLM-generated content at query #102
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception):
        with work_in_progress("Failing task"):
            raise Exception("Test error")


# LLM-generated content at query #103
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
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #104
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def task():
        time.sleep(0.1)

    task()


# LLM-generated content at query #105
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


# LLM-generated content at query #106
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def sample_function():
        time.sleep(0.1)

    sample_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #107
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


# LLM-generated content at query #108
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #109
#--------------------------

```python
def test_work_in_progress():
    # Test using context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test using decorator
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #110
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate work


# LLM-generated content at query #111
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #112
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #113
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work
    assert True  # Context manager completes without error


# LLM-generated content at query #114
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


# LLM-generated content at query #115
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


# LLM-generated content at query #116
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
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


# LLM-generated content at query #117
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


# LLM-generated content at query #118
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
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #119
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


# LLM-generated content at query #120
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #121
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()

    with work_in_progress() as wip:
        time.sleep(0.1)
    assert wip is None


# LLM-generated content at query #122
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert wip is None

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert wip is None

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
        return "result"

    result = test_func()
    assert result == "result"


# LLM-generated content at query #123
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


# LLM-generated content at query #124
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as wip:
        time.sleep(0.1)  # Simulate some work
    assert True  # Just checking it completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True

    # Test that output contains expected format
    import io
    import sys
    from unittest.mock import patch

    output = io.StringIO()
    with patch('sys.stdout', new=output):
        with work_in_progress("Test output"):
            time.sleep(0.1)
        output_str = output.getvalue()
        assert "Test output... done." in output_str
        assert "s)" in output_str  # Should contain time measurement


# LLM-generated content at query #125
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


# LLM-generated content at query #126
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #127
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()


# LLM-generated content at query #128
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test context manager behavior
    with pytest.raises(RuntimeError):
        with work_in_progress("Failing task"):
            raise RuntimeError("Test error")

    # Test decorator behavior
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #129
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert wip is None  # Context manager yields None

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


# LLM-generated content at query #130
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #131
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


# LLM-generated content at query #132
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    duration = time.time() - start
    assert 0.09 <= duration <= 0.11


# LLM-generated content at query #133
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


# LLM-generated content at query #134
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Context manager completed without errors

def test_work_in_progress_decorator():
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True  # Decorator completed without errors

def test_work_in_progress_output(capsys):
    with work_in_progress("Output test"):
        time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Output test... done." in captured.out
    assert "s)" in captured.out


# LLM-generated content at query #135
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)  # Simulate some work

    test_function()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #2
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator functionality
    @work_in_progress("Decorated task")
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
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #4
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
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
    # Test with default description
    with work_in_progress() as ctx:
        assert ctx is None  # Context manager yields None

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        assert ctx is None
        time.sleep(0.1)  # Small delay to test timing

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
        return "result"

    result = test_func()
    assert result == "result"


# LLM-generated content at query #6
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

    # Test timing accuracy (should be approximately 0.1 seconds)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15  # Allow some tolerance


# LLM-generated content at query #7
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    result = dummy_function()
    assert result is None


# LLM-generated content at query #8
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple task
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #9
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #10
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True  # Just checking it completes without error


# LLM-generated content at query #11
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #12
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
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #13
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #14
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test that the context manager completes without errors
    assert True


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception):
        with work_in_progress("Failing task"):
            raise Exception("Test error")

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #16
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorator test")
    def dummy_func():
        time.sleep(0.1)
    dummy_func()


# LLM-generated content at query #17
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #18
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #19
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(RuntimeError):
        with work_in_progress("Failing task") as _:
            raise RuntimeError("Test error")

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
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


# LLM-generated content at query #22
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

    # Test timing accuracy (should be close to 0.1s)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15  # Allow some margin for system delays


# LLM-generated content at query #23
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate some work

    dummy_function()


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

def test_work_in_progress_decorator():
    @work_in_progress("Decorator test")
    def dummy_func():
        time.sleep(0.1)
    dummy_func()


# LLM-generated content at query #25
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(AttributeError):
        with work_in_progress("Test task") as ctx:
            ctx.non_existent_attr

    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #26
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
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
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_progress():
    # Test as a context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test as a decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_task():
        time.sleep(0.1)  # Simulate work

    dummy_task()


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator functionality
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress() as _:
        time.sleep(0.1)


# LLM-generated content at query #33
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress() as _:
        time.sleep(0.1)


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    result = test_func()
    assert result is None


# LLM-generated content at query #36
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


# LLM-generated content at query #37
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


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)  # Simulate work
    assert True  # Context manager completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)  # Simulate work
    assert True  # Context manager completes without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()
    assert True  # Decorator completes without error


# LLM-generated content at query #39
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
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #40
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Context manager executed without errors


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #46
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


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True  # Just checking it completes without error


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #50
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #51
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


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
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


# LLM-generated content at query #53
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify it completes without error

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True  # Just to verify it completes without error


# LLM-generated content at query #54
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


# LLM-generated content at query #55
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #56
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #57
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
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #58
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #59
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

    # Test default description
    with work_in_progress() as wip:
        time.sleep(0.1)


# LLM-generated content at query #60
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #61
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test block") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Test decorator")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #62
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #63
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)
    assert isinstance(ctx, contextlib._GeneratorContextManager)


# LLM-generated content at query #64
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


# LLM-generated content at query #65
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #66
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #67
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #68
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        assert wip is None  # Context manager should yield None
    # Output should be "Work in progress... done. (X.XXs)"

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        assert wip is None
    # Output should be "Custom task... done. (X.XXs)"

    # Test timing accuracy (should be close to 0.1 seconds)
    start = time.time()
    with work_in_progress("Sleep test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15  # Allow some margin for system delays


# LLM-generated content at query #69
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True


# LLM-generated content at query #70
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test operation") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #71
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


# LLM-generated content at query #72
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #73
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

    # Test that the output contains the expected parts
    import io
    import sys

    output = io.StringIO()
    sys.stdout = output

    with work_in_progress("Custom task"):
        time.sleep(0.1)

    sys.stdout = sys.__stdout__
    assert "Custom task... done." in output.getvalue()


# LLM-generated content at query #74
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


# LLM-generated content at query #75
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()


# LLM-generated content at query #76
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate work


# LLM-generated content at query #77
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #78
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


# LLM-generated content at query #79
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #80
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #81
#--------------------------

```python
def test_work_in_progress():
    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    # Test with context manager
    with work_in_progress("Test context manager"):
        time.sleep(0.1)

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #82
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #83
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage
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
    end = time.time()

    assert abs((end - start) - 0.1) < 0.01


# LLM-generated content at query #84
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

    result = test_func()
    assert result is None


# LLM-generated content at query #85
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #86
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
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #87
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #88
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

    # Test with default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #89
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


# LLM-generated content at query #90
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate work


# LLM-generated content at query #91
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #92
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


# LLM-generated content at query #93
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #94
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


# LLM-generated content at query #95
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert wip is None

def test_work_in_progress_with_decorator():
    @work_in_progress("Decorated test")
    def dummy_func():
        time.sleep(0.1)
    dummy_func()


# LLM-generated content at query #96
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

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #97
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple operation
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #98
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #99
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


# LLM-generated content at query #100
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task"):
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #101
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def task():
        time.sleep(0.1)

    task()


# LLM-generated content at query #102
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)

    # Test decorator functionality
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #103
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Check if the context manager completes without errors


# LLM-generated content at query #104
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

    # Test timing is reasonable (should be close to 0.1s)
    start = time.time()
    with work_in_progress():
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15


# LLM-generated content at query #105
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception):
        with work_in_progress("Failing task"):
            raise Exception("Task failed")

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #106
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


# LLM-generated content at query #107
#--------------------------

```python
def test_work_in_progress():
    # Test as a context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test as a decorator
    @work_in_progress("Test decorator")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #108
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just checking it completes without error

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True  # Just checking it completes without error


# LLM-generated content at query #109
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #110
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
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


# LLM-generated content at query #111
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
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #112
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just checking it runs without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()
    assert True


# LLM-generated content at query #113
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #114
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as cm:
        time.sleep(0.1)
    assert cm is None  # Context manager doesn't return anything

    # Test with custom description
    with work_in_progress("Custom task") as cm:
        time.sleep(0.1)
    assert cm is None

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test timing output (approximate)
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.1 <= elapsed <= 0.2  # Allow some margin for execution overhead


# LLM-generated content at query #115
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #116
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress() as _:
        time.sleep(0.1)


# LLM-generated content at query #117
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()


# LLM-generated content at query #118
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
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


# LLM-generated content at query #119
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

    # Test timing accuracy
    start = time.time()
    with work_in_progress("Timing test"):
        time.sleep(0.2)
    end = time.time()
    assert 0.15 <= (end - start) <= 0.25


# LLM-generated content at query #120
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #121
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


# LLM-generated content at query #122
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


# LLM-generated content at query #123
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #124
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate work


# LLM-generated content at query #125
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorator test")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #126
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
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #127
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as wip:
        time.sleep(0.1)  # Simulate some work

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test as decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test that the output contains expected strings
    import io
    import sys

    output = io.StringIO()
    sys.stdout = output

    with work_in_progress("Test output") as wip:
        time.sleep(0.1)

    sys.stdout = sys.__stdout__

    assert "Test output... done." in output.getvalue()
    assert "s)" in output.getvalue()  # Check for time format


# LLM-generated content at query #128
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with default description
    with work_in_progress() as ctx:
        time.sleep(0.1)

    # Test custom description
    with work_in_progress("Custom task") as ctx:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test timing accuracy (should be at least 0.1 seconds)
    start = time.time()
    with work_in_progress() as ctx:
        time.sleep(0.1)
    elapsed = time.time() - start
    assert elapsed >= 0.1


# LLM-generated content at query #129
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


# LLM-generated content at query #130
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #131
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


# LLM-generated content at query #132
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception) as excinfo:
        with work_in_progress("Failing task"):
            raise ValueError("Test error")
    assert "Test error" in str(excinfo.value)

    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #133
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    with pytest.raises(Exception) as excinfo:
        with work_in_progress("Failing task"):
            raise ValueError("Test error")
    assert "Test error" in str(excinfo.value)

    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()


# LLM-generated content at query #134
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #135
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


# LLM-generated content at query #136
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


# LLM-generated content at query #137
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as ctx:
        time.sleep(0.1)
    assert ctx is None  # Context manager should not return anything

    # Test with custom description
    with work_in_progress("Custom task") as ctx:
        time.sleep(0.1)
    assert ctx is None

    # Test as decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)
    test_func()

    # Test timing is reasonable (should be ~0.1s)
    start = time.time()
    with work_in_progress():
        time.sleep(0.1)
    elapsed = time.time() - start
    assert 0.05 <= elapsed <= 0.15


# LLM-generated content at query #138
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #139
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_function():
        time.sleep(0.1)  # Simulate work

    test_function()


# LLM-generated content at query #140
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


# LLM-generated content at query #141
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


# LLM-generated content at query #142
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #143
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator
    @work_in_progress("Decorated task")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


# LLM-generated content at query #144
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


# LLM-generated content at query #145
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


# LLM-generated content at query #146
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #147
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to ensure the test passes if no exceptions occur


# LLM-generated content at query #148
#--------------------------

```python
def test_work_in_progress():
    # Test basic functionality with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #149
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)  # Simulate work


# LLM-generated content at query #150
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #151
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


# LLM-generated content at query #152
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate some work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #153
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)  # Simulate work

    test_func()


# LLM-generated content at query #154
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test operation") as wip:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #155
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test with decorator
    @work_in_progress("Decorated function")
    def dummy_function():
        time.sleep(0.1)  # Simulate work

    dummy_function()


# LLM-generated content at query #156
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # The context manager should complete without errors


# LLM-generated content at query #157
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
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


# LLM-generated content at query #158
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


# LLM-generated content at query #159
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #160
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #161
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes

    # Test as a decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True  # Just to verify the decorator completes


# LLM-generated content at query #162
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
    with work_in_progress("Test task") as cm:
        time.sleep(0.1)  # Simulate work

    # Test the decorator usage
    @work_in_progress("Decorated function")
    def test_function():
        time.sleep(0.1)

    test_function()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #163
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


# LLM-generated content at query #164
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated task")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #165
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


# LLM-generated content at query #166
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)
    assert True  # Just to check if the context manager completes without error

    # Test with custom description
    with work_in_progress("Custom task") as wip:
        time.sleep(0.1)
    assert True  # Just to check if the context manager completes without error

    # Test as a decorator
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()
    assert True  # Just to check if the decorator completes without error


# LLM-generated content at query #167
#--------------------------

```python
def test_work_in_progress():
    # Test basic usage with context manager
    with work_in_progress("Test task") as ctx:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated function")
    def test_func():
        time.sleep(0.1)

    test_func()

    # Test default description
    with work_in_progress():
        time.sleep(0.1)


# LLM-generated content at query #168
#--------------------------

```python
def test_work_in_progress():
    # Test context manager usage
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)  # Simulate work

    # Test decorator usage
    @work_in_progress("Decorated task")
    def task():
        time.sleep(0.1)  # Simulate work

    task()


# LLM-generated content at query #169
#--------------------------

```python
def test_work_in_progress():
    # Test with default description
    with work_in_progress() as wip:
        time.sleep(0.1)

    # Test with custom description
    with work_in_progress("Custom test") as wip:
        time.sleep(0.1)

    # Test as decorator
    @work_in_progress("Decorator test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #170
#--------------------------

```python
def test_work_in_progress():
    # Test the context manager with a simple block
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


# LLM-generated content at query #171
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


# LLM-generated content at query #172
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test task") as _:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Decorated test")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #173
#--------------------------

```python
def test_work_in_progress():
    # Test with context manager
    with work_in_progress("Test context") as wip:
        time.sleep(0.1)

    # Test with decorator
    @work_in_progress("Test decorator")
    def test_func():
        time.sleep(0.1)

    test_func()


# LLM-generated content at query #174
#--------------------------

```python
def test_work_in_progress():
    with work_in_progress("Test task") as wip:
        time.sleep(0.1)
    assert True  # Just to verify the context manager completes without error


