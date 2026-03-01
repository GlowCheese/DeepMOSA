####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert True  # If we get here, the KeyboardInterrupt was not captured

    # Test that other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    assert True  # If we get here, the ValueError was captured

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert True  # If we get here, the KeyboardInterrupt was captured

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    original_excepthook = sys.excepthook

    # Test KeyboardInterrupt is not captured
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None

    # Test KeyboardInterrupt is captured
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is skipped by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions trigger the IPython hook
    def raise_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when specified
    sys.excepthook = original_excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_default_handler(4, 2) == 2

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y, z=10):
        return x / y

    # Should not raise
    assert func_custom_handler(4, 2) == 2

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def gen_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    assert list(gen_func(4, 2)) == [2, 8]

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        list(gen_func(4, 0))

    # Test handler argument validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, x, y=1):
        pass

    def bad_handler4(e, z):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)(lambda: None)

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)(lambda x, y: None)

    with pytest.raises(ValueError, match="Argument 'y' in exception handler does not match"):
        exception_wrapper(bad_handler3)(lambda x: None)

    with pytest.raises(ValueError, match="Argument 'z' matches wrapped method argument"):
        exception_wrapper(bad_handler4)(lambda z: None)


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that exception in generator is handled
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test with custom handler and generator
    handler_called = False
    handler_args = None

    def custom_handler_gen(e, x, y):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y)

    @exception_wrapper(custom_handler_gen)
    def generator_func_custom(x, y):
        yield x / y
        yield x * y

    gen = generator_func_custom(1, 0)
    with pytest.raises(StopIteration):
        next(gen)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0

    # Test with handler that has default arguments
    def handler_with_defaults(e, x, y, default_arg=10):
        return e, x, y, default_arg

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        return x / y

    result = func_with_defaults(1, 0)
    assert isinstance(result[0], ZeroDivisionError)
    assert result[1] == 1
    assert result[2] == 0
    assert result[3] == 10

    # Test with handler that has **kwargs
    def handler_with_kwargs(e, x, y, **kwargs):
        return e, x, y, kwargs

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y, z=3):
        return x / y

    result = func_with_kwargs(1, 0, z=5)
    assert isinstance(result[0], ZeroDivisionError)
    assert result[1] == 1
    assert result[2] == 0
    assert result[3] == {"z": 5}

    # Test that handler without exception argument raises ValueError
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exc_arg():
            pass

    # Test that handler with *args raises ValueError
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs():
            pass

    # Test that handler with matching argument with default raises ValueError
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default(x):
            pass


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(10, 2) == 5

    # Test that exception is handled
    try:
        func_with_default_handler(10, 0)
    except ZeroDivisionError:
        assert False, "Exception should have been handled"

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Test that custom handler is called with correct arguments
    try:
        func_with_custom_handler(10, 0, z=5)
    except ZeroDivisionError:
        assert False, "Exception should have been handled"

    assert handler_called
    e, x, y, kwargs = handler_args
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {"z": 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("Test error")
        yield x + 1

    gen = generator_func(5)
    assert next(gen) == 5
    try:
        next(gen)
    except ValueError:
        assert False, "Exception should have been handled"

    # Test that exception is raised if handler raises
    def raising_handler(e):
        raise RuntimeError("Handler error")

    @exception_wrapper(raising_handler)
    def func_with_raising_handler():
        raise ValueError("Original error")

    with pytest.raises(RuntimeError, match="Handler error"):
        func_with_raising_handler()

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func_with_bad_handler():
            pass

    # Test with invalid handler (varargs)
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs_handler():
            pass

    # Test with invalid handler (argument doesn't match)
    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match"):
        @exception_wrapper(lambda e, z: None)
        def func_with_mismatched_handler(x):
            pass

    # Test with invalid handler (default value for matched argument)
    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_handler(x):
            pass


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should log exception but not raise
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(6, 2) == 3

    # Should call handler but not raise
    func_with_custom_handler(1, 0, z=5)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        yield 1 / x

    gen = generator_func(2)
    assert next(gen) == 2
    # Should log exception but not raise
    next(gen)

    # Test handler argument validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, x=1):
        pass

    def bad_handler4(e, x):
        pass

    def func_for_bad_handlers(x, y):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler1)(func_for_bad_handlers)

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler2)(func_for_bad_handlers)

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler3)(func_for_bad_handlers)

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler4)(func_for_bad_handlers)


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_error()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args("value1", "value2", extra="extra_value")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "value1"
    assert handler_args[2] == "value2"
    assert handler_args[3] == {"extra": "extra_value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator error")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler argument validation
    def invalid_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(invalid_handler)
        def func_with_mismatch():
            pass

    def handler_with_defaults(e, arg_with_default="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_matching_default_arg(arg_with_default):
            pass


# LLM-generated content at query #8
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise KeyError("custom error")

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    # Test default handler
    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    func_custom(1, 2, extra_kw="value")
    assert handler_called
    assert isinstance(handler_args[0], KeyError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "value"}}

    # Test generator
    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test that non-exception cases work
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default_arg(arg1):
            pass


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", custom_kwarg="test_value")

    # Test exception during logging
    with pytest.raises(Exception):
        try:
            raise Exception("Original error")
        except Exception as e:
            # Mock log to raise an exception
            original_log = log
            def mock_log(*args, **kwargs):
                raise Exception("Logging error")
            log = mock_log
            log_exception(e)
            log = original_log


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def test_func_default():
        raise ValueError("Test error")

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def test_func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise TypeError("Custom error")

    # Test with generator function
    @exception_wrapper()
    def test_generator():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    # Test default handler
    with pytest.raises(ValueError):
        test_func_default()

    # Test custom handler
    test_func_custom(1, 2, extra_kw="value")
    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == "default"
    assert handler_args[4] == {"extra_kw": "value"}

    # Test generator
    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test that non-exception cases work normally
    @exception_wrapper()
    def test_normal():
        return "success"

    assert test_normal() == "success"

    @exception_wrapper()
    def test_normal_generator():
        yield "a"
        yield "b"

    gen = test_normal_generator()
    assert next(gen) == "a"
    assert next(gen) == "b"


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0, z=4)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {"z": 4}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that exception in generator is handled
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test successful execution
    @exception_wrapper()
    def func_success(x, y):
        return x + y

    assert func_success(1, 2) == 3

    # Test that handler with invalid arguments raises ValueError
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, invalid_arg: None)
        def func_invalid_handler(x, y):
            return x / y

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x, y, default_arg=1: None)
        def func_default_arg(x, y):
            return x / y


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test default behavior (KeyboardInterrupt not captured)
    mocker.patch('sys.excepthook')
    mocker.patch('IPython.core.ultratb.FormattedTB')
    register_ipython_excepthook()

    # Simulate KeyboardInterrupt
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    sys.__excepthook__.assert_called_once_with(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Simulate other exception
    sys.excepthook(ValueError, ValueError(), None)
    ultratb.FormattedTB.assert_called_once()

    # Reset mocks
    sys.excepthook.reset_mock()
    sys.__excepthook__.reset_mock()
    ultratb.FormattedTB.reset_mock()

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    # Simulate KeyboardInterrupt (should now be captured)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    ultratb.FormattedTB.assert_called_once()


# LLM-generated content at query #14
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="value")


# LLM-generated content at query #15
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    # This should not trigger the IPython debugger
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    # This should trigger the IPython debugger
    try:
        raise_value_error()
    except ValueError:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass

    # Restore the original exception hook
    sys.excepthook = original_excepthook


# LLM-generated content at query #16
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="value")


# LLM-generated content at query #17
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="test_value")


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=4)

    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 4}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_called = False

    def custom_handler_gen(e, x, **kwargs):
        nonlocal handler_called
        handler_called = True

    @exception_wrapper(custom_handler_gen)
    def generator_func_custom(x, y):
        yield x / y

    gen = generator_func_custom(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    assert handler_called

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    assert func_no_exception(1, 2) == 3

    # Test with generator that doesn't raise
    @exception_wrapper()
    def generator_no_exception(x, y):
        yield x + y
        yield x * y

    gen = generator_no_exception(2, 3)
    assert next(gen) == 5
    assert next(gen) == 6


# LLM-generated content at query #19
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    original_excepthook = sys.excepthook
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass
    assert sys.excepthook == original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook

    # Test that BdbQuit is not captured
    def raise_bdb_quit():
        raise BdbQuit()

    original_excepthook = sys.excepthook
    try:
        raise_bdb_quit()
    except BdbQuit:
        pass
    assert sys.excepthook == original_excepthook

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    original_excepthook = sys.excepthook
    try:
        raise_value_error()
    except ValueError:
        pass
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="value")


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 20}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2, z=3) == 2
    assert list(generator_func(4, 2)) == [2, 8]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matching arguments
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            return x

    # Test handler with varargs
    def varargs_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(varargs_handler)
        def varargs_func(x):
            return x


# LLM-generated content at query #22
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Test type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Test runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(4, 2) == 2

    # Test that exception is handled
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test that the function works normally
    assert func_with_custom_handler(4, 2) == 2

    # Test that exception is handled with custom handler
    func_with_custom_handler(4, 0, z=20)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that the generator works normally
    gen = generator_func(4, 2)
    assert next(gen) == 2
    assert next(gen) == 8

    # Test that exception in generator is handled
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_func_custom(x, y):
        yield x / y
        yield x * y

    gen = generator_func_custom(4, 0)
    next(gen)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {}

    # Test handler with args that don't match wrapped function
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with default values for matching args
    def another_bad_handler(e, x=10):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(another_bad_handler)
        def func_with_another_bad_handler(x, y):
            return x / y


# LLM-generated content at query #24
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_hook(type, value, traceback):
        raise AssertionError("KeyboardInterrupt should not be captured")

    sys.excepthook = test_hook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Expected

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except BdbQuit:
        pass  # Expected when captured

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        @exception_wrapper(lambda: None)
        def func():
            pass

    # Test with invalid handler (varargs)
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    # Test with handler argument not matching wrapped function
    with pytest.raises(ValueError, match="Argument 'invalid_arg' in exception handler does not match"):
        @exception_wrapper(lambda e, invalid_arg: None)
        def func():
            pass

    # Test with handler argument with default matching wrapped function
    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, x=1: None)
        def func(x):
            pass

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    func_with_custom_handler(10, 2, z=5)
    assert len(handler_calls) == 0  # No exception

    func_with_custom_handler(10, 0, z=5)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    gen = generator_func(10, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    def bad_handler1():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)

    def bad_handler2(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)

    def bad_handler3(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler"):
        exception_wrapper(bad_handler3)

    def bad_handler4(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler4)


# LLM-generated content at query #27
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2.0

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise and not call handler
    assert func_with_custom_handler(4, 2) == 2.0
    assert len(handler_calls) == 0

    # Should call handler with correct arguments
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0, z=20)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8.0

    # Should handle exception in generator
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Argument 'nonexistent' in exception handler"):
        @exception_wrapper(lambda e, nonexistent: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        @exception_wrapper(lambda e, x=1: None)
        def func(x):
            pass


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(10, 2) == 5

    # Test that exception is caught and handled
    try:
        func_with_default_handler(10, 0)
    except ZeroDivisionError:
        assert False, "Exception should have been caught by the wrapper"

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3, *args, **kwargs):
        return x / y

    # Test that the custom handler is called with correct arguments
    try:
        func_with_custom_handler(10, 0, z=5, a=1, b=2)
    except ZeroDivisionError:
        assert False, "Exception should have been caught by the wrapper"

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 10
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 5, "args": (1, 2), "kwargs": {"a": 1, "b": 2}}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that the generator works normally
    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    # Test that exception in generator is caught
    gen = generator_func(10, 0)
    try:
        next(gen)
    except ZeroDivisionError:
        assert False, "Exception should have been caught by the wrapper"

    # Test that the generator is exhausted after exception
    with pytest.raises(StopIteration):
        next(gen)

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        @exception_wrapper(lambda: None)
        def func():
            pass

    # Test with invalid handler (varargs)
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    # Test with invalid handler (argument mismatch)
    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, z: None)
        def func(x, y):
            pass

    # Test with invalid handler (default value on matching argument)
    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, x=1: None)
        def func(x, y):
            pass


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=4)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 4}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_called = False

    def custom_handler_gen(e, x, **kwargs):
        nonlocal handler_called
        handler_called = True

    @exception_wrapper(custom_handler_gen)
    def generator_func_custom(x, y):
        yield x / y

    gen = generator_func_custom(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    assert handler_called

    # Test successful execution
    @exception_wrapper()
    def func_success(x, y):
        return x + y

    assert func_success(1, 2) == 3

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_bad_handler(x, y):
            return x / y

    # Test with handler having default values for matching args
    def bad_handler_defaults(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler_defaults)
        def func_bad_handler_defaults(x, y):
            return x / y


# LLM-generated content at query #30
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Test message", extra_arg="value")


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(4, 2) == 2

    # Test that exception is handled
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))
        raise ValueError("Handled")

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test that custom handler is called with correct arguments
    with pytest.raises(ValueError, match="Handled"):
        func_with_custom_handler(1, 0, z=20)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("Generator error")
        yield x + 1

    gen = generator_func(5)
    assert next(gen) == 5
    with pytest.raises(ValueError, match="Generator error"):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_with_custom_handler(x):
        yield x
        raise ValueError("Generator error")
        yield x + 1

    gen = generator_with_custom_handler(5)
    assert next(gen) == 5
    with pytest.raises(ValueError, match="Handled"):
        next(gen)

    assert len(handler_calls) == 1
    e, x, kwargs = handler_calls[0]
    assert isinstance(e, ValueError)
    assert x == 5
    assert kwargs == {}

    # Test handler argument validation
    def bad_handler1():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)(lambda: None)

    def bad_handler2(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)(lambda: None)

    def bad_handler3(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler"):
        exception_wrapper(bad_handler3)(lambda x: x)

    def bad_handler4(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler4)(lambda x: x)


# LLM-generated content at query #32
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="value")


# LLM-generated content at query #33
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    mock_ipython_hook = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_sys_excepthook = mocker.patch('sys.excepthook')

    # Call the function to register the hook
    register_ipython_excepthook()

    # Verify that the IPython hook was created with the correct parameters
    mock_ipython_hook.assert_called_once_with(mode='Context', color_scheme='Linux', call_pdb=1)

    # Verify that the sys.excepthook was set
    assert mock_sys_excepthook.called

    # Test that KeyboardInterrupt is not captured by default
    mock_excepthook = mock_sys_excepthook
    mock_excepthook.side_effect = None

    # Simulate a KeyboardInterrupt
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        excepthook = sys.excepthook
        excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Verify that the original excepthook was called for KeyboardInterrupt
    assert sys.__excepthook__ == excepthook

    # Test that other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        excepthook = sys.excepthook
        excepthook(ValueError, ValueError("Test exception"), None)

    # Verify that the IPython hook was called for ValueError
    mock_ipython_hook.return_value.assert_called_once()

    # Reset the sys.excepthook to its original value
    sys.excepthook = sys.__excepthook__

    # Test that capture_keyboard_interrupt=True captures KeyboardInterrupt
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    # Simulate a KeyboardInterrupt
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        excepthook = sys.excepthook
        excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Verify that the IPython hook was called for KeyboardInterrupt
    mock_ipython_hook.return_value.assert_called()

    # Reset the sys.excepthook to its original value
    sys.excepthook = sys.__excepthook__


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x):
        return x / 0

    # Mock log_exception to verify it's called
    import unittest.mock
    with unittest.mock.patch('module.log_exception') as mock_log:
        func_with_default_handler(1)
        mock_log.assert_called_once()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {'e': e, 'x': x, 'kwargs': kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2):
        return x / 0

    func_with_custom_handler(1, y=3)
    assert handler_called
    assert isinstance(handler_args['e'], ZeroDivisionError)
    assert handler_args['x'] == 1
    assert handler_args['kwargs'] == {'y': 3}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        yield x / 0

    with unittest.mock.patch('module.log_exception') as mock_log:
        gen = generator_func(1)
        next(gen)  # First yield
        try:
            next(gen)  # Should raise ZeroDivisionError
        except StopIteration:
            pass
        mock_log.assert_called_once()

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x):
            return x

    # Test with handler having default values for matched args
    def handler_with_defaults(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_handler_defaults(x):
            return x

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func(x):
        return x * 2

    assert successful_func(5) == 10


# LLM-generated content at query #35
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="extra_value")

    # Test exception during logging
    with patch('builtins.print') as mock_print:
        with patch('traceback.format_exc', side_effect=Exception("Logging error")):
            try:
                raise ValueError("Original error")
            except ValueError as e:
                log_exception(e)
        assert mock_print.call_count == 2


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler with matching args
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        raise RuntimeError("handled")

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(RuntimeError, match="handled"):
        func_custom(1, 2)

    # Test custom handler with kwargs
    def handler_kwargs(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs["arg2"] == 2
        raise RuntimeError("handled")

    @exception_wrapper(handler_kwargs)
    def func_kwargs(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(RuntimeError, match="handled"):
        func_kwargs(1, 2)

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

    # Test invalid handler (no exception arg)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func_invalid():
            pass

    # Test invalid handler (varargs)
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func_varargs():
            pass

    # Test invalid handler (arg mismatch)
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(lambda e, nonexistent: None)
        def func_mismatch():
            pass

    # Test invalid handler (default value on matching arg)
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, arg1=None: None)
        def func_default_value(arg1):
            pass


# LLM-generated content at query #37
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_arg="extra_value")


# LLM-generated content at query #38
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with a custom exception
    class CustomError(Exception):
        pass

    try:
        raise CustomError("Custom error message")
    except CustomError as e:
        log_exception(e, "Another custom message", extra_arg="value")

    # Test that log_exception raises the logging exception if it occurs
    with pytest.raises(Exception):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            # Mock log to raise an exception
            original_log = log
            log = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Logging error"))
            log_exception(e)
            log = original_log


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        return x / y

    func_with_custom_handler(1, 0)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with default arguments
    handler_called = False

    def handler_with_defaults(e, x, y, default_arg=10):
        nonlocal handler_called
        handler_called = True
        assert x == 1
        assert y == 0
        assert default_arg == 10

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        return x / y

    func_with_defaults(1, 0)
    assert handler_called

    # Test handler with **kwargs
    handler_called = False

    def handler_with_kwargs(e, x, **kwargs):
        nonlocal handler_called
        handler_called = True
        assert x == 1
        assert kwargs == {'y': 0, 'z': 2}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y, z):
        return x / y

    func_with_kwargs(1, 0, z=2)
    assert handler_called

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3

    # Test handler with mismatched arguments (should raise ValueError)
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with default values for matched arguments (should raise ValueError)
    def bad_handler_with_defaults(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler_with_defaults)
        def func_with_bad_handler_defaults(x, y):
            return x / y


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    func_with_default_handler(4, 2)

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise
    func_with_custom_handler(4, 2)

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2
    assert next(gen) == 8

    # Should log exception
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with varargs
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs_handler(x, y):
            return x / y

    # Test handler with no exception argument
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func_with_no_exc_handler(x, y):
            return x / y

    # Test handler with mismatched arguments
    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match"):
        @exception_wrapper(lambda e, z: None)
        def func_with_mismatched_handler(x, y):
            return x / y

    # Test handler with default values on matched arguments
    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_handler(x, y):
            return x / y


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_exception()

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default"):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler("value1", "value2", extra_kwarg="extra")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['arg2'] == "value2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kwarg": "extra"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test handler with default values on matching args
    def another_bad_handler(e, arg1="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(another_bad_handler)
        def func_with_another_bad_handler(arg1):
            pass


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2.0

    # Should log exception but not raise
    func_with_default_handler(1, 0)

    # Test custom handler
    handler_calls = []

    def custom_handler(e, x, y, z=None, **kwargs):
        handler_calls.append((e, x, y, z, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10, **kwargs):
        return x / y

    # Should not raise
    assert func_with_custom_handler(10, 2, z=5) == 5.0

    # Should call handler but not raise
    func_with_custom_handler(1, 0, z=5, extra="value")
    assert len(handler_calls) == 1
    e, x, y, z, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert z == 5
    assert kwargs == {"extra": "value"}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8.0

    # Should log exception but not raise
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_bad_handler(x):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_in_handler(x):
            pass

    # Test handler with no exception argument
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_with_no_e_handler(x):
            pass


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2=None):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2=None):
        raise TypeError("custom error")

    func_custom("value1", arg2="value2")
    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == "value1"
    assert handler_args[2] == "value2"

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent: None)
        def func_mismatch():
            pass

    # Test handler with default values on matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=None: None)
        def func_defaults(arg1):
            pass

    # Test handler without exception argument
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exc():
            pass

    # Test handler with *args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_varargs():
            pass


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_exception()

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler("val1", "val2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kw": "extra"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatch():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_with_default_mismatch(arg1):
            pass

    # Test handler with no exception argument
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exception_arg():
            pass


# LLM-generated content at query #45
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_arg="extra_value")


# LLM-generated content at query #46
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be caught"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        assert False, "KeyboardInterrupt should be caught"
    else:
        pass

    # Test that other exceptions are captured
    register_ipython_excepthook()
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be caught"

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0, z=20)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that exception in generator is handled
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test with custom handler and generator
    handler_called = False
    handler_args = None

    def custom_handler_gen(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler_gen)
    def generator_func_custom(x, y):
        yield x / y
        yield x * y

    gen = generator_func_custom(1, 0)
    with pytest.raises(StopIteration):
        next(gen)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {}

    # Test that function works normally when no exception is raised
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    assert func_no_exception(1, 2) == 3

    # Test with custom handler and no exception
    handler_called = False

    @exception_wrapper(custom_handler)
    def func_no_exception_custom(x, y):
        return x + y

    assert func_no_exception_custom(1, 2) == 3
    assert not handler_called


# LLM-generated content at query #48
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_arg="extra_value")


# LLM-generated content at query #49
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {"z": 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    list(generator_func(4, 2)) == [2, 8]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matching arguments
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            return x

    # Test handler with varargs
    def varargs_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(varargs_handler)
        def varargs_func(x):
            return x


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            pass

    # Test handler with default values on matched arguments
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            pass


# LLM-generated content at query #52
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_keyboard_interrupt():
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Should not enter IPython debugger
    test_keyboard_interrupt()

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    def test_keyboard_interrupt_captured():
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Should enter IPython debugger (but we can't test that directly)
    test_keyboard_interrupt_captured()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #53
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook

    # Call the function to register the new excepthook
    register_ipython_excepthook()

    # Verify that the excepthook has been changed
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_excepthook(type, value, traceback):
        assert type is not KeyboardInterrupt

    # Simulate a KeyboardInterrupt
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit
    except BdbQuit:
        sys.excepthook(BdbQuit, BdbQuit(), None)

    # Test that other exceptions are captured
    def test_other_exceptions(type, value, traceback):
        assert type is ValueError

    # Simulate a ValueError
    try:
        raise ValueError("test")
    except ValueError:
        sys.excepthook(ValueError, ValueError("test"), None)

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #54
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2, 8]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matching args
    def another_bad_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(another_bad_handler)
        def another_bad_func(x):
            return x


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test exception")

    with pytest.raises(ValueError):
        func_with_exception()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler("val1", "val2", extra_kwarg="extra")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kwarg": "extra"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def invalid_handler_func():
            pass

    with pytest.raises(ValueError):
        def handler_with_varargs(e, *args):
            pass

        @exception_wrapper(handler_with_varargs)
        def invalid_varargs_func():
            pass

    with pytest.raises(ValueError):
        def handler_with_default(e, arg="default"):
            pass

        @exception_wrapper(handler_with_default)
        def func_with_matching_arg(arg):
            pass


# LLM-generated content at query #56
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom error")

    with pytest.raises(RuntimeError):
        func_custom("val1", "val2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "val1"
    assert handler_args[2] == "val2"
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "extra"}}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("Generator error")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        exception_wrapper(bad_handler)(lambda: None)

    # Test with handler having default values for matched args
    def bad_handler2(e, arg1="default"):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        exception_wrapper(bad_handler2)(lambda arg1: None)


# LLM-generated content at query #57
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    log_exception(error)

    # Test subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime message", extra="info")


# LLM-generated content at query #58
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        tb = sys.exc_info()[2]
        sys.excepthook(type, KeyboardInterrupt(), tb)
        assert True  # If we get here, the exception was not captured

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        tb = sys.exc_info()[2]
        # This should not raise an exception, indicating it was captured
        sys.excepthook(type, KeyboardInterrupt(), tb)
        assert True

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #59
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    register_ipython_excepthook()
    assert sys.excepthook is not None

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    original_excepthook = sys.excepthook
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    # The excepthook should not have been called for KeyboardInterrupt

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    # The excepthook should have been called for KeyboardInterrupt

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit
    except BdbQuit:
        pass
    # The excepthook should not have been called for BdbQuit

    # Test that other exceptions are captured
    register_ipython_excepthook()
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    # The excepthook should have been called for ValueError


# LLM-generated content at query #60
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    # Should not trigger IPython debugger
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    # Should trigger IPython debugger (we can't test the actual debugger, but we can check the hook is called)
    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #61
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0, z=5)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {"z": 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x + y

    # Test that exception in generator is handled
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test that successful execution returns correct value
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2, 6]

    # Test with mismatched handler arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatch(x):
            return x

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_in_handler(x):
            return x


# LLM-generated content at query #62
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    mock_ultratb = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_excepthook = mocker.patch('sys.excepthook')

    # Call the function
    register_ipython_excepthook()

    # Assert that the excepthook was set
    mock_excepthook.assert_called_once()

    # Test that KeyboardInterrupt is skipped by default
    excepthook = sys.excepthook
    test_exception = KeyboardInterrupt()
    excepthook(type(test_exception), test_exception, None)
    mock_excepthook.assert_called_with(type(test_exception), test_exception, None)

    # Test that other exceptions are handled by IPython
    test_exception = ValueError("test")
    excepthook(type(test_exception), test_exception, None)
    mock_ultratb.return_value.assert_called_once_with(type(test_exception), test_exception, None)

    # Reset sys.excepthook
    sys.excepthook = mock_excepthook

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    excepthook = sys.excepthook
    test_exception = KeyboardInterrupt()
    excepthook(type(test_exception), test_exception, None)
    mock_ultratb.return_value.assert_called_with(type(test_exception), test_exception, None)


# LLM-generated content at query #63
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    register_ipython_excepthook()
    assert sys.excepthook is not None

    # Test that KeyboardInterrupt is not captured by default
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not original_excepthook

    # Test that BdbQuit is always skipped
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that the excepthook is called with the correct arguments
    def mock_excepthook(type, value, traceback):
        assert type is ValueError
        assert isinstance(value, ValueError)
        assert isinstance(traceback, traceback)

    sys.excepthook = mock_excepthook
    register_ipython_excepthook()
    try:
        raise ValueError("test")
    except:
        pass

    # Test that the excepthook is called with the correct arguments when KeyboardInterrupt is captured
    def mock_excepthook_ki(type, value, traceback):
        assert type is KeyboardInterrupt
        assert isinstance(value, KeyboardInterrupt)
        assert isinstance(traceback, traceback)

    sys.excepthook = mock_excepthook_ki
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except:
        pass

    # Test that the excepthook is not called when KeyboardInterrupt is not captured
    def mock_excepthook_no_ki(type, value, traceback):
        assert False, "This should not be called"

    sys.excepthook = mock_excepthook_no_ki
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except:
        pass

    # Test that the excepthook is not called when BdbQuit is raised
    def mock_excepthook_no_bdbquit(type, value, traceback):
        assert False, "This should not be called"

    sys.excepthook = mock_excepthook_no_bdbquit
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except:
        pass


# LLM-generated content at query #64
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="extra_value")


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_exception()

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler(1, 2, extra_kw="value")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == 1
    assert handler_args['arg2'] == 2
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {'extra_kw': 'value'}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test handler with default values on matching args
    def handler_with_defaults(e, arg1="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_matching_defaults(arg1):
            pass


# LLM-generated content at query #66
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    assert sys.excepthook is not None

    # Test that the excepthook is not the default one
    assert sys.excepthook is not sys.__excepthook__

    # Test that the excepthook is the one we registered
    assert sys.excepthook is excepthook

    # Test that the excepthook is called when an exception is raised
    try:
        raise ValueError("test")
    except ValueError:
        pass
    else:
        assert False, "Exception was not raised"

    # Test that the excepthook is not called when a KeyboardInterrupt is raised
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "Exception was not raised"

    # Test that the excepthook is called when a KeyboardInterrupt is raised and capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "Exception was not raised"

    # Test that the excepthook is not called when a BdbQuit is raised
    try:
        raise BdbQuit
    except BdbQuit:
        pass
    else:
        assert False, "Exception was not raised"


# LLM-generated content at query #67
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="extra_value")


# LLM-generated content at query #68
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with custom message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="value")


# LLM-generated content at query #69
#--------------------------

```python
def test_log_exception():
    # Test with a generic exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with no user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", level="critical", extra_info="additional")


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that normal execution works
    assert func_with_default_handler(10, 2) == 5

    # Test that exception is handled
    try:
        func_with_default_handler(10, 0)
    except ZeroDivisionError:
        pytest.fail("Exception should have been caught by wrapper")

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Test normal execution
    assert func_with_custom_handler(10, 2) == 5

    # Test exception handling
    try:
        func_with_custom_handler(10, 0, z=5)
    except ZeroDivisionError:
        pytest.fail("Exception should have been caught by wrapper")

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test normal execution
    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    # Test exception handling in generator
    try:
        gen = generator_func(10, 0)
        next(gen)
    except ZeroDivisionError:
        pytest.fail("Exception should have been caught by wrapper")

    # Test with varargs
    def handler_with_varargs(e, x, **kwargs):
        handler_calls.append((e, x, kwargs))

    @exception_wrapper(handler_with_varargs)
    def func_with_varargs(x, *args, **kwargs):
        return x / args[0]

    # Test normal execution
    assert func_with_varargs(10, 2, extra=5) == 5

    # Test exception handling
    try:
        func_with_varargs(10, 0, extra=5)
    except ZeroDivisionError:
        pytest.fail("Exception should have been caught by wrapper")

    assert len(handler_calls) == 2  # Previous call + this one
    e, x, kwargs = handler_calls[-1]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert kwargs == {'args': (0,), 'extra': 5}

    # Test handler validation
    def bad_handler1(e):
        pass

    def bad_handler2(e, x, y=1):
        pass

    def bad_handler3(e, *args):
        pass

    def bad_handler4(e, z):
        pass

    def func_to_wrap(x, y):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)(func_to_wrap)

    with pytest.raises(ValueError, match="Argument 'y' in exception handler does not match"):
        exception_wrapper(bad_handler2)(func_to_wrap)

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler3)(func_to_wrap)

    with pytest.raises(ValueError, match="Argument 'z' matches wrapped method argument"):
        exception_wrapper(bad_handler4)(func_to_wrap)


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise and not call handler
    assert func_with_custom_handler(4, 2) == 2
    assert len(handler_calls) == 0

    # Should call handler with correct arguments
    func_with_custom_handler(4, 0, z=20)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2
    assert next(gen) == 8

    # Should handle exception in generator
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, x=1):
        pass

    def bad_handler4(e, z):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler3)

    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match"):
        exception_wrapper(bad_handler4)


# LLM-generated content at query #72
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    result = func_with_custom_handler(10, 2)
    assert result == 5
    assert len(handler_calls) == 0

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(10, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 3}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    gen = generator_func(10, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    def bad_handler1():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)

    def bad_handler2(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)

    def bad_handler3(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler"):
        exception_wrapper(bad_handler3)

    def bad_handler4(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler4)


# LLM-generated content at query #73
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be captured"

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be captured"

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be captured"

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should be captured when capture_keyboard_interrupt is True"

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #74
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_default_handler(4, 2) == 2.0

    # Should log exception
    func_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y, z=3):
        return x / y

    # Should not raise and call handler
    func_custom_handler(1, 0)
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ZeroDivisionError)
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 0
    assert handler_calls[0][3] == {'z': 3}

    # Test with generator function
    @exception_wrapper()
    def gen_func(x):
        yield x
        raise ValueError("test error")
        yield x + 1

    gen = gen_func(10)
    assert next(gen) == 10
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test with invalid handler
    try:
        @exception_wrapper(lambda: None)
        def invalid_handler_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)

    try:
        def handler_with_defaults(e, x=1):
            pass

        @exception_wrapper(handler_with_defaults)
        def func_with_matching_arg(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #75
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that the function skips KeyboardInterrupt by default
    def test_hook(type, value, traceback):
        assert type is not KeyboardInterrupt
    sys.excepthook = test_hook
    register_ipython_excepthook()
    try:
        raise KeyboardInterrupt
    except:
        pass

    # Test that the function captures KeyboardInterrupt when specified
    def test_hook_capture(type, value, traceback):
        assert type is KeyboardInterrupt
    sys.excepthook = test_hook_capture
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #76
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exc_arg():
            pass

    # Test with invalid handler (varargs)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs():
            pass

    # Test with invalid handler (matching arg with default)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default(x):
            pass

    # Test with invalid handler (non-matching arg without default)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, z: None)
        def func_without_z(x, y):
            pass

    # Test successful execution
    @exception_wrapper()
    def func_success(x, y):
        return x + y

    assert func_success(1, 2) == 3

    # Test successful generator execution
    @exception_wrapper()
    def generator_success(x, y):
        yield x + y
        yield x * y

    gen = generator_success(1, 2)
    assert next(gen) == 3
    assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(gen)


# LLM-generated content at query #77
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_arg="extra_value")


# LLM-generated content at query #78
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should not enter IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Test that capture_keyboard_interrupt=True captures KeyboardInterrupt
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #79
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_error()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args(1, 2, extra_kw="value")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "value"}}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("Generator test")

    with pytest.raises(StopIteration):
        list(gen_func())

    # Test handler with mismatched arguments
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func():
            pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, arg_with_default=None: None)
        def func(arg_with_default):
            pass


# LLM-generated content at query #80
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    sys.excepthook = original_excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when specified
    sys.excepthook = original_excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #81
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error message", extra_arg="extra_value")


# LLM-generated content at query #82
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    original_excepthook = sys.excepthook

    # Test KeyboardInterrupt is not captured
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should not enter IPython debugger

    # Test other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook is not original_excepthook

    # Test KeyboardInterrupt is captured
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Test BdbQuit is never captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass  # Should not enter IPython debugger


# LLM-generated content at query #83
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_exception(type, value, traceback):
        assert type is not KeyboardInterrupt

    sys.excepthook = test_exception
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    captured = False

    def test_exception_capture(type, value, traceback):
        nonlocal captured
        if type is KeyboardInterrupt:
            captured = True

    sys.excepthook = test_exception_capture
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert captured

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #84
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #85
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the excepthook correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be caught by default"

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be caught"

    # Test that other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be caught"

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #86
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should not trigger IPython debugger

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should trigger IPython debugger (but we can't test that directly)

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except BdbQuit:
        pass  # Should not trigger IPython debugger

    # Test that other exceptions trigger IPython debugger
    register_ipython_excepthook()
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass  # Should trigger IPython debugger (but we can't test that directly)

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #87
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Test message", extra_arg="extra_value")


# LLM-generated content at query #88
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    original_excepthook = sys.excepthook

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook

    # Test that BdbQuit is always skipped
    test_exception = BdbQuit
    try:
        raise test_exception
    except:
        sys.excepthook(type, value, traceback)

    # Test that KeyboardInterrupt is skipped when capture_keyboard_interrupt=False
    test_exception = KeyboardInterrupt
    try:
        raise test_exception
    except:
        sys.excepthook(type, value, traceback)

    # Test that other exceptions are handled by IPython
    test_exception = ValueError("test")
    try:
        raise test_exception
    except:
        # This should trigger the IPython debugger
        # We can't easily test this without mocking IPython, but we can verify the hook is set
        assert sys.excepthook is not None

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #89
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="extra_value")


# LLM-generated content at query #90
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("test")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #91
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be caught"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        assert False, "KeyboardInterrupt should be caught"
    else:
        pass

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be caught"

    # Test that other exceptions are captured
    register_ipython_excepthook()
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be caught"

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #92
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise exception
    assert func_with_default_handler(4, 2) == 2.0

    # Should handle exception
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise exception
    assert func_with_custom_handler(4, 2) == 2.0

    # Should call custom handler
    func_with_custom_handler(1, 0, z=20)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should not raise exception
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8

    # Should handle exception in generator
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with varargs (should raise ValueError)
    def bad_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler)

    # Test handler with no exception argument (should raise ValueError)
    def another_bad_handler(x, y):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(another_bad_handler)

    # Test handler with default values for matched arguments (should raise ValueError)
    def yet_another_bad_handler(e, x=1, y=2):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(yet_another_bad_handler)


# LLM-generated content at query #93
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_hook(type, value, traceback):
        assert type is not KeyboardInterrupt

    sys.excepthook = test_hook
    try:
        raise KeyboardInterrupt()
    except:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    sys.excepthook = test_hook
    try:
        raise KeyboardInterrupt()
    except:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #94
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_error()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default"):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler("val1", "val2", default_arg="custom_default", extra="extra_val")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "val1"
    assert handler_args[2] == "val2"
    assert handler_args[3] == "custom_default"
    assert handler_args[4] == {"extra": "extra_val"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Argument 'missing_arg' in exception handler"):
        @exception_wrapper(lambda e, missing_arg: None)
        def func():
            pass

    with pytest.raises(ValueError, match="Argument 'default_arg' matches wrapped method argument"):
        @exception_wrapper(lambda e, default_arg="default": None)
        def func(default_arg):
            pass


# LLM-generated content at query #95
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be caught"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        assert False, "KeyboardInterrupt should be caught"
    else:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #96
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the excepthook correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    # Restore the original excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    # Restore the original excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    # Restore the original excepthook
    sys.excepthook = original_excepthook

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #97
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test subprocess.CalledProcessError without output
    error_no_output = subprocess.CalledProcessError(1, "cmd")
    log_exception(error_no_output)

    # Test exception during logging (should print and re-raise)
    class BadException(Exception):
        def __str__(self):
            raise RuntimeError("Can't convert to string")

    try:
        raise BadException()
    except BadException as e:
        with pytest.raises(RuntimeError):
            log_exception(e)


# LLM-generated content at query #98
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(4, 2) == 2.0

    # Test that exception is caught and logged
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10, *args, **kwargs):
        return x / y

    # Test that the function works normally
    assert func_with_custom_handler(4, 2) == 2.0

    # Test that custom handler is called with correct arguments
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0, z=20, a=1, b=2)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 4
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 20, "args": (1, 2), "kwargs": {"a": 1, "b": 2}}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that the generator works normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8

    # Test that exception in generator is caught
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test invalid handler signatures
    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        @exception_wrapper(lambda: None)
        def func1():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func2():
            pass

    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, z: None)
        def func3(x, y):
            pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, x=1: None)
        def func4(x, y):
            pass


# LLM-generated content at query #99
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="extra_value")


# LLM-generated content at query #100
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_arg="extra_value")


# LLM-generated content at query #101
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None

    # Test that the hook is called with the correct arguments
    def mock_hook(type, value, traceback):
        mock_hook.called = True
        mock_hook.type = type
        mock_hook.value = value
        mock_hook.traceback = traceback

    mock_hook.called = False
    original_hook = sys.excepthook
    sys.excepthook = mock_hook

    try:
        raise ValueError("test")
    except ValueError:
        exc_info = sys.exc_info()
        sys.excepthook(*exc_info)

    assert mock_hook.called
    assert mock_hook.type is ValueError
    assert str(mock_hook.value) == "test"
    assert mock_hook.traceback is not None

    # Restore original hook
    sys.excepthook = original_hook


# LLM-generated content at query #102
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, **kwargs):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, extra_arg="value")

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"extra_arg": "value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    with pytest.raises(ZeroDivisionError):
        list(generator_func(1, 0))

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2]


# LLM-generated content at query #103
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_arg="extra_value")

    # Test exception during logging
    with pytest.raises(Exception):
        try:
            raise AttributeError("Attribute error")
        except AttributeError as e:
            # Mock log to raise an exception
            original_log = log
            log = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Log error"))
            try:
                log_exception(e)
            finally:
                log = original_log


# LLM-generated content at query #104
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #105
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception():
    # Test with a generic exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with no user message
    try:
        raise RuntimeError("Another test error")
    except RuntimeError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Type error occurred", level="critical")


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should not enter IPython debugger

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #4
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    mock_ultratb = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_ipython_hook = mock_ultratb.return_value

    # Call the function
    register_ipython_excepthook()

    # Check that sys.excepthook was set
    assert sys.excepthook is not None

    # Simulate an exception
    test_exception = ValueError("test")
    test_traceback = None

    # Call the excepthook with a non-skip exception
    sys.excepthook(type(test_exception), test_exception, test_traceback)

    # Check that the IPython hook was called
    mock_ipython_hook.assert_called_once_with(type(test_exception), test_exception, test_traceback)

    # Reset the excepthook
    sys.excepthook = sys.__excepthook__

    # Test with KeyboardInterrupt (should not be captured by default)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    ki_exception = KeyboardInterrupt()
    sys.excepthook(type(ki_exception), ki_exception, test_traceback)
    mock_ipython_hook.assert_called_once()  # Should not be called again

    # Test with KeyboardInterrupt capture enabled
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    sys.excepthook(type(ki_exception), ki_exception, test_traceback)
    mock_ipython_hook.assert_called_with(type(ki_exception), ki_exception, test_traceback)

    # Reset the excepthook
    sys.excepthook = sys.__excepthook__


# LLM-generated content at query #5
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    # Should not trigger IPython debugger
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        raise_value_error()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #6
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test subprocess.CalledProcessError without output
    error_no_output = subprocess.CalledProcessError(1, "cmd")
    log_exception(error_no_output)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime message", extra_param="value")


# LLM-generated content at query #7
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    # This should not trigger the IPython debugger
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    # Mock the IPython hook to avoid actually launching IPython
    with pytest.raises(ValueError):
        raise_value_error()

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        raise_keyboard_interrupt()

    # Restore the original exception hook
    sys.excepthook = original_excepthook


# LLM-generated content at query #8
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e, "Subprocess error")

    # Test with no custom message
    try:
        raise RuntimeError("Another error")
    except RuntimeError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Type error occurred", extra_kwarg="value")


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("custom error")

    with pytest.raises(RuntimeError):
        func_custom(1, 2, extra_kw="value")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "value"}}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator error")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matching args
    def bad_default_handler(e, arg1="default"):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        exception_wrapper(bad_default_handler)(lambda arg1: None)


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2.0

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise
    assert func_with_custom_handler(4, 2) == 2.0

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8

    # Should log exception in generator
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func1():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func2():
            pass

    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match"):
        @exception_wrapper(lambda e, z: None)
        def func3(x, y):
            pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        @exception_wrapper(lambda e, x=1: None)
        def func4(x):
            pass


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_default_handler(10, 2) == 5

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_custom_handler(10, 2) == 5
    assert len(handler_calls) == 0

    # Should handle exception
    func_custom_handler(10, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 3}

    # Test with generator
    @exception_wrapper()
    def gen_func(x):
        yield x
        yield 1 / 0  # This should be caught

    gen = gen_func(5)
    assert next(gen) == 5
    try:
        next(gen)
    except StopIteration:
        pass  # The exception was handled, generator stops

    # Test with invalid handler (no exception argument)
    try:
        @exception_wrapper(lambda: None)
        def func_invalid_handler():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

    # Test with handler having *args
    try:
        @exception_wrapper(lambda e, *args: None)
        def func_varargs_handler():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

    # Test with handler having default for matched argument
    try:
        @exception_wrapper(lambda e, x=1: None)
        def func_default_matched(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_excepthook(type, value, traceback):
        assert type is not KeyboardInterrupt

    sys.excepthook = test_excepthook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    def test_excepthook_capture(type, value, traceback):
        assert type is KeyboardInterrupt

    sys.excepthook = test_excepthook_capture
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    mock_ultratb = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_hook = mock_ultratb.return_value
    original_excepthook = sys.excepthook

    # Call the function
    register_ipython_excepthook()

    # Verify the excepthook was set
    assert sys.excepthook is not original_excepthook

    # Test the excepthook behavior
    test_exception = ValueError("test")
    try:
        raise test_exception
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()

    # Call the registered excepthook
    sys.excepthook(exc_type, exc_value, exc_traceback)

    # Verify the IPython hook was called
    mock_hook.assert_called_once_with(exc_type, exc_value, exc_traceback)

    # Test with KeyboardInterrupt (should not be captured)
    sys.excepthook = original_excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)

    try:
        raise KeyboardInterrupt()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()

    # Mock the original excepthook to verify it's called
    with mocker.patch.object(sys, '__excepthook__') as mock_original:
        sys.excepthook(exc_type, exc_value, exc_traceback)
        mock_original.assert_called_once_with(exc_type, exc_value, exc_traceback)
        mock_hook.assert_not_called()

    # Test with BdbQuit (should not be captured)
    try:
        raise BdbQuit()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()

    with mocker.patch.object(sys, '__excepthook__') as mock_original:
        sys.excepthook(exc_type, exc_value, exc_traceback)
        mock_original.assert_called_once_with(exc_type, exc_value, exc_traceback)
        mock_hook.assert_not_called()

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 5}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatch(x):
            pass

    # Test handler with default values for matched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default(x):
            pass

    # Test handler with varargs
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs(x):
            pass


# LLM-generated content at query #15
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should not trigger the IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass  # Should trigger the IPython debugger

    # Reset the excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should trigger the IPython debugger

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise
    assert func_with_custom_handler(4, 2) == 2

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("Test error")

    gen = generator_func(1)
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_with_custom_handler(x):
        yield x
        raise ValueError("Test error")

    gen = generator_with_custom_handler(1)
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)
    assert len(handler_calls) == 1
    e, x, kwargs = handler_calls[0]
    assert isinstance(e, ValueError)
    assert x == 1
    assert kwargs == {}

    # Test handler validation
    def bad_handler1():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)(lambda: None)

    def bad_handler2(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)(lambda: None)

    def bad_handler3(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' in exception handler does not match"):
        exception_wrapper(bad_handler3)(lambda y: None)

    def bad_handler4(e, x):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler4)(lambda x=1: None)


# LLM-generated content at query #17
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Custom type error")
    except Exception as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="Error output")
    except Exception as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except Exception as e:
        log_exception(e, extra_kwarg="extra_value")

    # Test exception during logging
    try:
        raise ValueError("Original error")
    except Exception as e:
        # Mock log to raise an exception
        original_log = log
        def mock_log(*args, **kwargs):
            raise RuntimeError("Log error")
        log = mock_log
        try:
            log_exception(e)
        finally:
            log = original_log


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    func_with_custom_handler(10, 2)
    assert len(handler_calls) == 0

    func_with_custom_handler(10, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 3}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    gen = generator_func(10, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler argument validation
    def invalid_handler1():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(invalid_handler1)

    def invalid_handler2(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(invalid_handler2)

    def invalid_handler3(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler"):
        exception_wrapper(invalid_handler3)

    def invalid_handler4(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(invalid_handler4)


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, **kwargs):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, extra_arg="value")

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"extra_arg": "value"}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    with pytest.raises(ZeroDivisionError):
        gen = generator_func(1, 0)
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matching args
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            return x


# LLM-generated content at query #20
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    original_hook = sys.excepthook

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook != original_hook

    # Test that BdbQuit is always skipped
    class TestException(Exception):
        pass

    # Mock the IPython hook to verify it's called
    called_with = []
    def mock_ipython_hook(type, value, traceback):
        called_with.append((type, value, traceback))

    # Replace the real IPython hook with our mock
    from IPython.core import ultratb
    original_ultra = ultratb.FormattedTB
    ultratb.FormattedTB = lambda *args, **kwargs: mock_ipython_hook

    try:
        # Test with regular exception
        try:
            raise TestException("test")
        except:
            sys.excepthook(*sys.exc_info())
        assert len(called_with) == 1
        assert called_with[0][0] is TestException

        # Test with KeyboardInterrupt (not captured by default)
        called_with.clear()
        try:
            raise KeyboardInterrupt()
        except:
            sys.excepthook(*sys.exc_info())
        assert len(called_with) == 0

        # Test with BdbQuit (always skipped)
        try:
            raise BdbQuit()
        except:
            sys.excepthook(*sys.exc_info())
        assert len(called_with) == 0

    finally:
        # Restore the original IPython hook
        ultratb.FormattedTB = original_ultra
        sys.excepthook = original_hook


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg="default", **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            "e": e,
            "arg1": arg1,
            "arg2": arg2,
            "default_arg": default_arg,
            "kwargs": kwargs
        }

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("custom error")

    with pytest.raises(RuntimeError):
        func_custom(1, 2, extra_kw="value")

    assert handler_called
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["default_arg"] == "default"
    assert handler_args["kwargs"] == {"extra_kw": "value"}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator error")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func_no_args():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        def handler_varargs(e, *args):
            pass

        @exception_wrapper(handler_varargs)
        def func_varargs():
            pass

    with pytest.raises(ValueError, match="Argument 'nonexistent' in exception handler"):
        def handler_bad_arg(e, nonexistent):
            pass

        @exception_wrapper(handler_bad_arg)
        def func_bad_arg():
            pass

    with pytest.raises(ValueError, match="Argument 'arg1' matches wrapped method argument"):
        def handler_default_arg(e, arg1="default"):
            pass

        @exception_wrapper(handler_default_arg)
        def func_default_arg(arg1):
            pass


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, **kwargs):
        raise TypeError("custom error")

    func_custom(1, 2, extra=3)
    assert handler_called
    assert handler_args[0].args[0] == "custom error"
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == {"kwargs": {"extra": 3}}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent: None)
        def func_mismatch():
            pass

    # Test handler with default values on matched args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=1: None)
        def func_defaults(arg1):
            pass

    # Test handler with varargs
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_varargs():
            pass

    # Test handler with no exception arg
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exc():
            pass


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper():
    # Test basic exception handling with default handler
    @exception_wrapper()
    def func_raises():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises()

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, optional=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'optional': optional,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, optional="default", **kwargs):
        raise RuntimeError("Custom handler test")

    func_with_args("val1", "val2", optional="custom", extra="kwarg")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['optional'] == "custom"
    assert handler_args['kwargs'] == {"extra": "kwarg"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    # Test handler with default values on matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default(arg1):
            pass

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_success():
        return "success"

    assert func_success() == "success"


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0, z=5)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = generator_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test successful execution
    @exception_wrapper()
    def successful_func():
        return "success"

    assert successful_func() == "success"

    # Test with generator that completes successfully
    @exception_wrapper()
    def successful_generator():
        yield 1
        yield 2

    assert list(successful_generator()) == [1, 2]


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3, **kwargs):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=4, extra="value")

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 4, "extra": "value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler does not match"):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with default values on matched arguments
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(bad_default_handler)
        def func_with_bad_default_handler(x, y):
            return x / y

    # Test handler with no exception argument
    def no_exception_handler():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        @exception_wrapper(no_exception_handler)
        def func_with_no_exception_handler(x, y):
            return x / y

    # Test handler with *args
    def varargs_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(varargs_handler)
        def func_with_varargs_handler(x, y):
            return x / y


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    handler_calls = []

    def custom_handler(e, arg1, arg2=None, **kwargs):
        handler_calls.append((e, arg1, arg2, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2=None, **kwargs):
        raise TypeError("custom error")

    with pytest.raises(TypeError):
        func_custom(1, arg2=2, extra=3)

    assert len(handler_calls) == 1
    e, arg1, arg2, kwargs = handler_calls[0]
    assert isinstance(e, TypeError)
    assert arg1 == 1
    assert arg2 == 2
    assert kwargs == {"extra": 3}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matching args
    def bad_handler2(e, existing_arg="default"):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        exception_wrapper(bad_handler2)(lambda existing_arg: None)


# LLM-generated content at query #27
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is handled
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        return x / y

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that exception in generator is handled
    gen = generator_func(1, 0)
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test with custom handler and generator
    handler_called = False

    @exception_wrapper(custom_handler)
    def generator_func_custom(x, y):
        yield x / y
        yield x * y

    gen = generator_func_custom(1, 0)
    try:
        next(gen)
    except StopIteration:
        pass
    assert handler_called

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    try:
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

    # Test with handler argument that has default value
    def handler_with_default(e, x, y=0):
        pass

    try:
        @exception_wrapper(handler_with_default)
        def func_with_handler_default(x, y):
            return x / y
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    result = func_no_exception(1, 2)
    assert result == 3


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x):
        return x / 0

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2, *args, z=None, **kwargs):
        return x / 0

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, y=3, z=4, w=5)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["kwargs"] == {"y": 3, "z": 4, "kwargs": {"w": 5}}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        yield 2 / 0
        yield 3

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, non_existent_arg: None)
        def func_with_mismatched_args(x):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_in_handler(x):
            pass

    # Test handler with no exception argument
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_with_no_exception_arg():
            pass

    # Test handler with *args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs_handler(x):
            pass


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2.0

    # Should handle exception
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(6, 2) == 3.0

    # Should handle exception
    func_with_custom_handler(1, 0, z=5)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8.0

    # Should handle exception in generator
    gen = generator_func(1, 0)
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test with varargs in handler (should raise)
    def bad_handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(bad_handler_with_varargs)
        def func_with_bad_handler(x):
            return x
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)

    # Test with non-matching argument (should raise)
    def bad_handler_with_non_matching_arg(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(bad_handler_with_non_matching_arg)
        def func_with_bad_handler(x):
            return x
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

    # Test with matching argument with default (should raise)
    def bad_handler_with_default(e, x=1):
        pass

    try:
        @exception_wrapper(bad_handler_with_default)
        def func_with_bad_handler(x):
            return x
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args("val1", "val2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "val1"
    assert handler_args[2] == "val2"
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "extra"}}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("Generator test")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched args
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default(arg1):
            pass


# LLM-generated content at query #31
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt()

    # This should not trigger the IPython debugger
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass

    # Test that other exceptions are captured
    def raise_value_error():
        raise ValueError("Test error")

    # This should trigger the IPython debugger
    try:
        raise_value_error()
    except ValueError:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise_keyboard_interrupt()
    except KeyboardInterrupt:
        pass

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Should not raise
    assert func_with_custom_handler(4, 2) == 2

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(4, 2)
    assert next(gen) == 2
    assert next(gen) == 8

    # Should log exception in generator
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with varargs in handler (should raise ValueError)
    def bad_handler_with_varargs(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler_with_varargs)

    # Test with no exception argument (should raise ValueError)
    def bad_handler_no_exception_arg():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler_no_exception_arg)

    # Test with handler argument not matching wrapped function
    def bad_handler_mismatch(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler"):
        exception_wrapper(bad_handler_mismatch)(lambda x: x)

    # Test with handler argument matching wrapped function but having default
    def bad_handler_with_default(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler_with_default)(lambda x: x)


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y=2):
        return x / y

    # Test that the function works normally
    assert func_with_default_handler(4, 2) == 2.0

    # Test that exception is handled
    try:
        func_with_default_handler(1, 0)
    except Exception:
        pytest.fail("Exception should have been handled by the wrapper")

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y=2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2, **kwargs):
        return x / y

    # Test that the function works normally
    assert func_with_custom_handler(4, 2) == 2.0

    # Test that custom handler is called with correct arguments
    try:
        func_with_custom_handler(1, 0, extra_arg="value")
    except Exception:
        pytest.fail("Exception should have been handled by the wrapper")

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"extra_arg": "value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y=2):
        yield x / y
        yield x * y

    # Test that the generator works normally
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8.0

    # Test that exception in generator is handled
    try:
        gen = generator_func(1, 0)
        next(gen)
    except Exception:
        pytest.fail("Exception in generator should have been handled by the wrapper")

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        @exception_wrapper(lambda: None)
        def func_with_invalid_handler():
            pass

    # Test with handler that has *args
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs_handler():
            pass

    # Test with handler that has mismatched arguments
    with pytest.raises(ValueError, match="Argument 'z' in exception handler does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, z: None)
        def func_with_mismatched_handler(x, y=2):
            pass

    # Test with handler that has default values for matched arguments
    with pytest.raises(ValueError, match="Argument 'y' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, x, y=2: None)
        def func_with_default_values_handler(x, y=2):
            pass


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(10, 2) == 5

    # Should handle exception
    func_with_default_handler(10, 0)

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(10, 2) == 5

    # Should call custom handler
    func_with_custom_handler(10, 0, z=5)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 10
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    # Should handle exception in generator
    gen = generator_func(10, 0)
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test with invalid handler (no exception argument)
    try:
        @exception_wrapper(lambda: None)
        def invalid_handler_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

    # Test with handler having *args
    try:
        @exception_wrapper(lambda e, *args: None)
        def varargs_handler_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

    # Test with handler having matching argument with default
    try:
        @exception_wrapper(lambda e, x=1: None)
        def default_arg_handler_func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "matches wrapped method argument, thus cannot have default values" in str(e)


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)
    assert handler_called
    assert handler_args[0].__class__ == ZeroDivisionError
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {"z": 20}

    # Test generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("Test error")

    gen = generator_func(5)
    assert next(gen) == 5
    with pytest.raises(ValueError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(10, 2) == 5
    assert func_with_custom_handler(10, 2) == 5

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matching args
    def bad_handler2(e, x=1):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler2)(lambda x: None)


# LLM-generated content at query #36
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test default behavior (KeyboardInterrupt not captured)
    mocker.patch('sys.excepthook')
    mocker.patch('IPython.core.ultratb.FormattedTB')

    register_ipython_excepthook()

    # Trigger exception
    try:
        raise ValueError("test")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    # Verify IPython hook was called
    assert sys.excepthook.called

    # Test KeyboardInterrupt behavior
    sys.excepthook.reset_mock()
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())

    # Verify original excepthook was called for KeyboardInterrupt
    assert sys.excepthook.called
    assert sys.excepthook.call_args[0][0] == KeyboardInterrupt

    # Test with capture_keyboard_interrupt=True
    sys.excepthook.reset_mock()
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())

    # Verify IPython hook was called for KeyboardInterrupt
    assert sys.excepthook.called


# LLM-generated content at query #37
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError and output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error message", extra_kwarg="extra_value")


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, arg1, arg2, kwarg1=None, **kwargs):
        handler_calls.append((e, arg1, arg2, kwarg1, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, kwarg1=None, **kwargs):
        raise TypeError("custom error")

    func_custom(1, 2, kwarg1=3, extra=4)
    assert len(handler_calls) == 1
    e, arg1, arg2, kwarg1, kwargs = handler_calls[0]
    assert isinstance(e, TypeError)
    assert arg1 == 1
    assert arg2 == 2
    assert kwarg1 == 3
    assert kwargs == {"extra": 4}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_bad():
            pass

    # Test with handler having default for existing argument
    def bad_handler2(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler2)
        def func_bad2(arg1):
            pass

    # Test with handler having varargs
    def bad_handler3(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler3)
        def func_bad3():
            pass

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_success():
        return "success"

    assert func_success() == "success"


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 5}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3

    # Test handler with default arguments
    def handler_with_defaults(e, x, default_arg="default", **kwargs):
        handler_calls.append((e, x, default_arg, kwargs))

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_defaults(1, 0)

    assert len(handler_calls) == 2
    e, x, default_arg, kwargs = handler_calls[1]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert default_arg == "default"
    assert kwargs == {"y": 0}

    # Test handler with mismatched arguments
    def bad_handler(e, non_existent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with varargs
    def handler_with_varargs(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_varargs)
        def func_with_varargs_handler(x, y):
            return x / y


# LLM-generated content at query #40
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    test_exception = KeyboardInterrupt()
    sys.excepthook(type(test_exception), test_exception, None)

    # Test that BdbQuit is not captured
    test_exception = BdbQuit()
    sys.excepthook(type(test_exception), test_exception, None)

    # Test that other exceptions are captured
    test_exception = ValueError("test")
    try:
        sys.excepthook(type(test_exception), test_exception, None)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Reset excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test with default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)

    assert handler_called
    e, x, y, kwargs = handler_args
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 20}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(4, 2) == 2
    assert func_with_custom_handler(4, 2) == 2
    assert list(generator_func(4, 2)) == [2, 8]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matched arguments
    def bad_default_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            return x


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_error()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, **kwargs):
        raise TypeError("Custom handler test")

    func_with_custom_handler(1, "test", extra="value")

    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == 1
    assert handler_args[2] == "test"
    assert handler_args[3] == {"extra": "value"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test handler with default values on matching args
    def handler_with_defaults(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_defaults(arg1):
            pass

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func():
        return "success"

    assert successful_func() == "success"

    # Test handler with varargs (should fail)
    def handler_with_varargs(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_varargs)
        def func_with_varargs_handler():
            pass


# LLM-generated content at query #43
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")


# LLM-generated content at query #44
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should not enter IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Reset to original excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Should enter IPython debugger (but we can't test that directly)

    # Reset to original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #45
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")

    # Test exception during logging
    with patch('builtins.print') as mock_print:
        with patch('traceback.format_exc', side_effect=Exception("Logging error")):
            try:
                raise ValueError("Original error")
            except ValueError as e:
                log_exception(e)
                assert mock_print.call_count == 2


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise TypeError("custom error")

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    # Test default handler
    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    func_custom(1, 2, extra_kw="extra")
    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == "default"
    assert handler_args[4] == {"extra_kw": "extra"}

    # Test generator function
    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)


# LLM-generated content at query #47
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Another test error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "test_cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test that log_exception raises the logging exception if it occurs
    with pytest.raises(RuntimeError):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            # Mock log to raise an exception
            original_log = log
            def mock_log(*args, **kwargs):
                raise RuntimeError("Logging error")
            log = mock_log
            try:
                log_exception(e)
            finally:
                log = original_log


# LLM-generated content at query #48
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    mock_ultratb = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_ipython_hook = mock_ultratb.return_value

    # Call the function
    register_ipython_excepthook()

    # Assert that the excepthook was set
    assert sys.excepthook is not None

    # Test with KeyboardInterrupt not captured
    test_exception = ValueError("test")
    mock_traceback = mocker.MagicMock()

    # Call the excepthook with a non-skip exception
    sys.excepthook(type(test_exception), test_exception, mock_traceback)
    mock_ipython_hook.assert_called_once_with(type(test_exception), test_exception, mock_traceback)

    # Reset the mock
    mock_ipython_hook.reset_mock()

    # Call the excepthook with a KeyboardInterrupt (should not be captured)
    ki_exception = KeyboardInterrupt()
    sys.excepthook(type(ki_exception), ki_exception, mock_traceback)
    mock_ipython_hook.assert_not_called()

    # Test with KeyboardInterrupt captured
    mock_ipython_hook.reset_mock()
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    # Call the excepthook with a KeyboardInterrupt (should be captured)
    sys.excepthook(type(ki_exception), ki_exception, mock_traceback)
    mock_ipython_hook.assert_called_once_with(type(ki_exception), ki_exception, mock_traceback)


# LLM-generated content at query #49
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_error()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args("val1", "val2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "val1"
    assert handler_args[2] == "val2"
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "extra"}}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    # Test handler with default values on matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default_on_match(arg1):
            pass


# LLM-generated content at query #50
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be captured"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should be captured when capture_keyboard_interrupt=True"

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be captured"

    # Test that other exceptions are captured
    register_ipython_excepthook()
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be captured"

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #51
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="value")


# LLM-generated content at query #52
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError and output
    error = subprocess.CalledProcessError(1, "cmd", output="Error output")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test exception")

    with pytest.raises(ValueError):
        func_with_exception()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler("value1", "value2", extra_kwarg="extra")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['arg2'] == "value2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kwarg": "extra"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_invalid_handler():
            pass

    with pytest.raises(ValueError):
        def handler_with_varargs(e, *args):
            pass

        @exception_wrapper(handler_with_varargs)
        def func_varargs_handler():
            pass

    with pytest.raises(ValueError):
        def handler_with_default(e, arg="default"):
            pass

        @exception_wrapper(handler_with_default)
        def func_default_handler(arg):
            pass


# LLM-generated content at query #54
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should not trigger IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass  # Should trigger IPython debugger (but we can't test that directly)

    # Reset the excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should trigger IPython debugger (but we can't test that directly)

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #55
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


# LLM-generated content at query #56
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    def test_excepthook(type, value, traceback):
        raise AssertionError("Should not be called for KeyboardInterrupt")

    sys.excepthook = test_excepthook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Expected

    # Test that other exceptions are captured
    try:
        raise ValueError("Test")
    except ValueError:
        pass  # Expected

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass  # Expected

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Expected

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom error")

    with pytest.raises(RuntimeError):
        func_custom("val1", "val2", extra_kw="extra")

    assert handler_called
    assert handler_args[0].args[0] == "Custom error"
    assert handler_args[1] == "val1"
    assert handler_args[2] == "val2"
    assert handler_args[3] == "default"
    assert handler_args[4] == {"kwargs": {"extra_kw": "extra"}}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("Generator error")

    with pytest.raises(StopIteration):
        list(gen_func())

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    # Test handler with default values on matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default_mismatch(arg1):
            pass


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(10, 2) == 5

    # Should handle exception
    func_with_default_handler(10, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(10, 2) == 5

    # Should call handler
    func_with_custom_handler(10, 0, z=5)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should work normally
    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    # Should handle exception in generator
    gen = generator_func(10, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler argument validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, x=1):
        pass

    def bad_handler4(e, z):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler1)
        def func1(x):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler2)
        def func2(x):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler3)
        def func3(x):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler4)
        def func4(x):
            pass


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_raises():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise TypeError("Custom handler test")

    func_with_args("value1", "value2", kw1="kwvalue1")
    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == "value1"
    assert handler_args[2] == "value2"
    assert handler_args[3] == {"kwargs": {"kw1": "kwvalue1"}}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test with mismatched handler arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    # Test with handler having default values for matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default(arg1):
            pass


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2=None, **kwargs):
        raise TypeError("custom error")

    func_custom(1, arg2=2, extra=3)
    assert handler_called
    assert isinstance(handler_args["e"], TypeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["kwargs"] == {"extra": 3}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched args
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matching args
    def bad_default_handler(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_default_handler)(lambda arg1: None)


# LLM-generated content at query #61
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_hook(type, value, traceback):
        raise AssertionError("KeyboardInterrupt should not be captured")

    sys.excepthook = test_hook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # Expected

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass  # Expected

    # Test that other exceptions are captured
    def test_hook_capture(type, value, traceback):
        assert type is ValueError
        assert str(value) == "test exception"

    sys.excepthook = test_hook_capture
    try:
        raise ValueError("test exception")
    except ValueError:
        pass  # Expected

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #62
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func1():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func1()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func2(arg1, arg2, default_arg="default", **kwargs):
        raise TypeError("Custom handler test")

    with pytest.raises(TypeError):
        func2("value1", "value2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args['e'], TypeError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['arg2'] == "value2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kw": "extra"}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def invalid_handler_func():
            pass

    with pytest.raises(ValueError):
        def handler_with_varargs(e, *args):
            pass

        @exception_wrapper(handler_with_varargs)
        def invalid_varargs_func():
            pass

    with pytest.raises(ValueError):
        def handler_with_default(e, arg="default"):
            pass

        @exception_wrapper(handler_with_default)
        def func_with_matching_arg(arg):
            pass


# LLM-generated content at query #63
#--------------------------

```python
def test_register_ipython_excepthook():
    # Save original excepthook
    original_excepthook = sys.excepthook

    # Test with capture_keyboard_interrupt=False
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not original_excepthook

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #64
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    handler_calls = []

    def custom_handler(e, x, y):
        handler_calls.append((e, x, y))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0)

    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ZeroDivisionError)
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 0

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with default arguments
    def handler_with_defaults(e, x, y, default_arg="default"):
        handler_calls.append((e, x, y, default_arg))

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_defaults(1, 0)

    assert len(handler_calls) == 2
    assert handler_calls[1][3] == "default"

    # Test handler with kwargs
    def handler_with_kwargs(e, x, **kwargs):
        handler_calls.append((e, x, kwargs))

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_kwargs(1, 0, z=4)

    assert len(handler_calls) == 3
    assert handler_calls[2][2] == {"y": 0, "z": 4}

    # Test successful execution
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3


# LLM-generated content at query #65
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook

    # Call the function to register the hook
    register_ipython_excepthook()

    # Verify that the excepthook has been replaced
    assert sys.excepthook is not original_excepthook

    # Test that the hook handles KeyboardInterrupt correctly by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        # The hook should not capture KeyboardInterrupt by default
        assert True

    # Test that the hook handles other exceptions by launching IPython
    try:
        raise ValueError("Test exception")
    except ValueError:
        # The hook should capture other exceptions
        assert True

    # Reset the excepthook
    sys.excepthook = original_excepthook

    # Test that the hook captures KeyboardInterrupt when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        # The hook should capture KeyboardInterrupt when capture_keyboard_interrupt is True
        assert True

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #66
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error occurred")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_arg="extra_value")


# LLM-generated content at query #67
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a user message
    try:
        raise TypeError("Another test error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test with subprocess.CalledProcessError
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_arg="extra_value")


# LLM-generated content at query #68
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        handler_calls.append((e, arg1, arg2, default_arg, kwargs))

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom error")

    func_custom(1, 2, extra_kw="value")

    assert len(handler_calls) == 1
    e, arg1, arg2, default_arg, kwargs = handler_calls[0]
    assert isinstance(e, RuntimeError)
    assert str(e) == "Custom error"
    assert arg1 == 1
    assert arg2 == 2
    assert default_arg == "default"
    assert kwargs == {"extra_kw": "value"}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("Generator error")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func_no_args():
            pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func_varargs():
            pass

    with pytest.raises(ValueError, match="Argument 'invalid' in exception handler"):
        @exception_wrapper(lambda e, invalid: None)
        def func_invalid_arg():
            pass

    with pytest.raises(ValueError, match="Argument 'arg1' matches wrapped method argument"):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default_conflict(arg1):
            pass


# LLM-generated content at query #69
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError and output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_arg="extra_value")


# LLM-generated content at query #70
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with no user message
    try:
        raise RuntimeError("Another test error")
    except RuntimeError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise TypeError("Type test error")
    except TypeError as e:
        log_exception(e, "Type error occurred", extra="info")


# LLM-generated content at query #71
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", extra_kwarg="extra_value")


# LLM-generated content at query #72
#--------------------------

```python
def test_log_exception():
    # Test with a simple exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test with subprocess.CalledProcessError and output
    try:
        raise subprocess.CalledProcessError(1, "test", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error message", extra_kwarg="extra_value")


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=20)

    assert handler_called
    assert handler_args[0].__class__ == ZeroDivisionError
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 20}

    # Test with generator
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test successful execution
    assert func_with_default_handler(10, 2) == 5
    assert func_with_custom_handler(10, 2, z=30) == 5
    assert list(generator_func(10, 2)) == [5.0, 20]

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matched arguments
    def bad_default_handler(e, x=10):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_default_handler)
        def bad_default_func(x):
            return x


# LLM-generated content at query #74
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook

    # Call the function
    register_ipython_excepthook()

    # Check that the excepthook has been replaced
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    def test_uncaught_exception():
        raise ValueError("Test exception")

    try:
        test_uncaught_exception()
    except ValueError:
        pass

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    def test_keyboard_interrupt():
        raise KeyboardInterrupt()

    try:
        test_keyboard_interrupt()
    except KeyboardInterrupt:
        pass

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #75
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y=2):
        return x / y

    # Test that function works normally
    assert func_with_default_handler(4, 2) == 2.0

    # Test that exception is handled (should not raise)
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y=2, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2, **kwargs):
        return x / y

    # Test that function works normally
    assert func_with_custom_handler(4, 2) == 2.0

    # Test that custom handler is called with correct arguments
    func_with_custom_handler(1, 0, extra="value")
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"extra": "value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("test error")
        yield x + 1

    gen = generator_func(1)
    assert next(gen) == 1
    # Should not raise, exception should be handled
    try:
        next(gen)
    except StopIteration:
        pass  # Expected after exception is handled

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler does not match"):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x):
            return x

    # Test handler with default values on matching arguments
    def another_bad_handler(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(another_bad_handler)
        def func_with_another_bad_handler(x):
            return x


# LLM-generated content at query #76
#--------------------------

```python
def test_register_ipython_excepthook(mocker):
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    mock_ultratb = mocker.patch('IPython.core.ultratb.FormattedTB')
    mock_hook = mock_ultratb.return_value
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    mock_hook.assert_not_called()

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    mock_hook.assert_called_once()

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    sys.excepthook(BdbQuit, BdbQuit(), None)
    mock_hook.assert_not_called()

    # Test that other exceptions are captured
    register_ipython_excepthook()
    sys.excepthook(ValueError, ValueError(), None)
    mock_hook.assert_called_once()


# LLM-generated content at query #77
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(10, 2) == 5

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, x, y, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(10, 2) == 5

    # Test exception handling
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(10, 0)

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(10, 0)

    assert handler_called
    e, x, y, kwargs = handler_args
    assert isinstance(e, ZeroDivisionError)
    assert x == 10
    assert y == 0
    assert kwargs == {'z': 3}

    # Test generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    gen = generator_func(10, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatched_args(x):
            return x

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_arg(x):
            return x


# LLM-generated content at query #78
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be caught"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except BdbQuit:
        pass
    else:
        assert True, "KeyboardInterrupt should be caught when capture_keyboard_interrupt=True"

    # Test that BdbQuit is always skipped
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be caught"

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #79
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler():
        raise ValueError("Test error")

    # Mock log_exception to verify it's called
    import unittest.mock
    with unittest.mock.patch('module.log_exception') as mock_log:
        func_with_default_handler()
        mock_log.assert_called_once()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, my_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'my_arg': my_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, my_arg=None, **kwargs):
        raise TypeError("Custom handler test")

    func_with_custom_handler(1, 2, my_arg=3, extra=4)
    assert handler_called
    assert isinstance(handler_args['e'], TypeError)
    assert handler_args['arg1'] == 1
    assert handler_args['arg2'] == 2
    assert handler_args['my_arg'] == 3
    assert handler_args['kwargs'] == {'extra': 4}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    with unittest.mock.patch('module.log_exception') as mock_log:
        gen = generator_func()
        list(gen)  # Consume generator to trigger exception
        mock_log.assert_called_once()

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func():
        return "success"

    result = successful_func()
    assert result == "success"

    # Test with generator that doesn't raise
    @exception_wrapper()
    def successful_generator():
        yield 1
        yield 2

    gen = successful_generator()
    assert list(gen) == [1, 2]

    # Test handler validation
    def bad_handler1():
        pass  # No exception argument

    def bad_handler2(e, *args):
        pass  # Has varargs

    def bad_handler3(e, missing_arg):
        pass  # Argument doesn't match wrapped function

    def bad_handler4(e, matching_arg=None):
        pass  # Matching argument has default

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler1)
        def test_func1():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler2)
        def test_func2():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler3)
        def test_func3(other_arg):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler4)
        def test_func4(matching_arg):
            pass


# LLM-generated content at query #80
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be captured by default"

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit
    except BdbQuit:
        pass
    else:
        assert False, "BdbQuit should not be captured"

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be captured"

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should be captured when capture_keyboard_interrupt is True"

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #81
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should handle exception
    func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise
    assert func_with_custom_handler(6, 2) == 3

    # Should call handler
    func_with_custom_handler(1, 0, z=5)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator
    @exception_wrapper()
    def generator_func(x):
        yield x
        yield 1 / x

    gen = generator_func(2)
    assert next(gen) == 2
    assert next(gen) == 0.5

    gen = generator_func(0)
    assert next(gen) == 0
    # Should handle exception in generator
    try:
        next(gen)
    except StopIteration:
        pass

    # Test handler argument validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, x=1):
        pass

    def bad_handler4(e, z):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler1)

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler2)

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler3)

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler4)
        def func(x, y):
            pass


# LLM-generated content at query #82
#--------------------------

```python
def test_log_exception():
    # Test with a basic exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with a custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom message")

    # Test with subprocess.CalledProcessError and output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error message", extra_kwarg="extra_value")


# LLM-generated content at query #83
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    original_excepthook = sys.excepthook

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook

    # Test that the excepthook is called with correct arguments
    def mock_ipython_hook(type, value, traceback):
        mock_ipython_hook.called = True
        mock_ipython_hook.args = (type, value, traceback)

    mock_ipython_hook.called = False
    mock_ipython_hook.args = None

    # Monkey patch the ipython_hook
    import importlib
    module = importlib.import_module('your_module_name')
    original_ipython_hook = module.ipython_hook
    module.ipython_hook = mock_ipython_hook

    try:
        # Trigger an exception
        try:
            raise ValueError("Test exception")
        except:
            sys.excepthook(*sys.exc_info())

        assert mock_ipython_hook.called
        assert mock_ipython_hook.args[0] == ValueError
        assert str(mock_ipython_hook.args[1]) == "Test exception"
        assert mock_ipython_hook.args[2] is not None
    finally:
        # Restore the original ipython_hook
        module.ipython_hook = original_ipython_hook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    mock_ipython_hook.called = False

    try:
        try:
            raise KeyboardInterrupt()
        except:
            sys.excepthook(*sys.exc_info())

        assert not mock_ipython_hook.called
    finally:
        module.ipython_hook = original_ipython_hook

    # Test that BdbQuit is not captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    mock_ipython_hook.called = False

    try:
        try:
            raise BdbQuit()
        except:
            sys.excepthook(*sys.exc_info())

        assert not mock_ipython_hook.called
    finally:
        module.ipython_hook = original_ipython_hook


# LLM-generated content at query #84
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test with custom user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom message")

    # Test subprocess.CalledProcessError with output
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Test message", extra_kwarg="value")


# LLM-generated content at query #85
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_raises()

    # Test custom handler with matching args
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args(1, "two", extra="value")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == 1
    assert handler_args[2] == "two"
    assert handler_args[3] == {"extra": "value"}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator test")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with invalid signature
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_invalid_handler():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_varargs_handler():
            pass

    # Test handler with matching args that have defaults
    def handler_with_defaults(e, arg1, arg2=None, **kwargs):
        pass

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(arg1, arg2, **kwargs):
        pass

    with pytest.raises(ValueError):
        func_with_defaults(1, 2)


# LLM-generated content at query #86
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises_exception():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_exception()

    # Test custom handler with matching arguments
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_args("value1", "value2", extra="kwarg")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "value1"
    assert handler_args[2] == "value2"
    assert handler_args[3] == {"extra": "kwarg"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with default values
    def handler_with_defaults(e, required_arg, default_arg="default"):
        return (e, required_arg, default_arg)

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(required_arg, other_arg):
        raise TypeError("Default values test")

    with pytest.raises(TypeError):
        result = func_with_defaults("req_val", "other_val")

    # Test invalid handler (no exception argument)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def invalid_handler_func():
            pass

    # Test invalid handler (varargs)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def invalid_varargs_handler():
            pass

    # Test invalid handler (matching arg with default)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_with_arg(arg1):
            pass


# LLM-generated content at query #87
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that the registered hook is a function
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Expected to be raised

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Expected to be raised (but should be caught by the hook)

    # Restore the original exception hook
    sys.excepthook = original_excepthook


# LLM-generated content at query #88
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_error()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("Custom handler test")

    func_with_args("value1", "value2", extra="extra_value")

    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "value1"
    assert handler_args[2] == "value2"
    assert handler_args[3] == {"extra": "extra_value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test with handler having default values for matching args
    def handler_with_defaults(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_matching_defaults(arg1):
            pass


# LLM-generated content at query #89
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, x, y, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "x": x, "y": y, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with default values on matching arguments
    def handler_with_defaults(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_matching_defaults(x):
            return x

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(2, 3) == 5

    # Test successful generator execution
    @exception_wrapper()
    def successful_generator(x, y):
        yield x
        yield y

    gen = successful_generator(2, 3)
    assert next(gen) == 2
    assert next(gen) == 3


# LLM-generated content at query #90
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Another test error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Test with kwargs")
    except RuntimeError as e:
        log_exception(e, "User message with kwargs", extra_param="value")


# LLM-generated content at query #91
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_default():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_default()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_custom(arg1, arg2, default_arg="default", extra_arg=None):
        raise RuntimeError("custom error")

    with pytest.raises(RuntimeError):
        func_custom("val1", "val2", extra_arg="extra")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {'extra_arg': 'extra'}

    # Test with generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    def bad_handler1():
        pass  # No exception argument

    def bad_handler2(e, *args):
        pass  # Has varargs

    def bad_handler3(e, missing_arg):
        pass  # Argument doesn't match wrapped function

    def bad_handler4(e, arg1="default"):
        pass  # Argument with default matches wrapped function

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler1)
        def func1():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler2)
        def func2():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler3)
        def func3(arg2):
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler4)
        def func4(arg1):
            pass


# LLM-generated content at query #92
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    assert sys.excepthook is not None

    # Test that KeyboardInterrupt is not captured by default
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert sys.excepthook is original_excepthook

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert sys.excepthook is not original_excepthook

    # Test that other exceptions are captured
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise ValueError("test")
    except ValueError:
        pass
    assert sys.excepthook is not original_excepthook


# LLM-generated content at query #93
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler():
        raise ValueError("Test error")

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, optional_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, optional_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, optional_arg="default", **kwargs):
        raise TypeError("Custom handler test")

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    # Test default handler
    with pytest.raises(ValueError):
        func_with_default_handler()

    # Test custom handler
    func_with_custom_handler("a", "b", optional_arg="custom", extra="kwarg")
    assert handler_called
    e, arg1, arg2, optional_arg, kwargs = handler_args
    assert isinstance(e, TypeError)
    assert arg1 == "a"
    assert arg2 == "b"
    assert optional_arg == "custom"
    assert kwargs == {"extra": "kwarg"}

    # Test generator function
    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test that non-generator functions work correctly
    @exception_wrapper()
    def normal_func():
        return "success"

    assert normal_func() == "success"

    # Test that handler with mismatched args raises ValueError
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test that handler with default values for matched args raises ValueError
    def bad_handler_with_defaults(e, arg1="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler_with_defaults)
        def func_with_bad_handler_defaults(arg1):
            pass


# LLM-generated content at query #94
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_error()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {
            'e': e,
            'arg1': arg1,
            'arg2': arg2,
            'default_arg': default_arg,
            'kwargs': kwargs
        }

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    with pytest.raises(RuntimeError):
        func_with_custom_handler(1, 2, extra_kw="value")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == 1
    assert handler_args['arg2'] == 2
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {'extra_kw': 'value'}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator error")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test handler with default values on matching args
    def handler_with_defaults_on_match(e, arg1="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults_on_match)
        def func_with_matching_default(arg1):
            pass


# LLM-generated content at query #95
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #96
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #97
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_func_custom(x, y):
        yield x / y
        yield x * y

    gen = generator_func_custom(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {}

    # Test successful execution (no exception)
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    result = successful_func(2, 3)
    assert result == 5

    # Test with custom handler and successful execution
    handler_calls.clear()
    result = successful_func(2, 3)
    assert result == 5
    assert len(handler_calls) == 0

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler does not match"):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matched arguments
    def another_bad_handler(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        exception_wrapper(another_bad_handler)(lambda x: None)

    # Test handler with varargs
    def varargs_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(varargs_handler)(lambda: None)

    # Test handler with no exception argument
    def no_exception_handler():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        exception_wrapper(no_exception_handler)(lambda: None)


# LLM-generated content at query #98
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {'z': 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_with_custom_handler(x, y):
        yield x / y
        yield x * y

    gen = generator_with_custom_handler(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {}

    # Test successful execution
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3

    # Test with mismatched handler arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test with handler having default values for matched args
    def another_bad_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(another_bad_handler)
        def func_with_another_bad_handler(x, y):
            return x / y


# LLM-generated content at query #99
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_with_default_handler()

    # Test with custom handler
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, **kwargs):
        raise TypeError("Custom handler test")

    with pytest.raises(TypeError):
        func_with_custom_handler("value1", "value2", extra="extra_value")

    assert handler_called
    assert isinstance(handler_args["e"], TypeError)
    assert handler_args["arg1"] == "value1"
    assert handler_args["arg2"] == "value2"
    assert handler_args["kwargs"] == {"extra": "extra_value"}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'nonexistent_arg' in exception handler does not match"):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler():
            pass

    # Test handler with default values on matching arguments
    def bad_default_handler(e, arg1="default"):
        pass

    with pytest.raises(ValueError, match="Argument 'arg1' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(bad_default_handler)
        def func_with_bad_default_handler(arg1):
            pass


# LLM-generated content at query #100
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that the function works normally when no exception is raised
    assert func_with_default_handler(4, 2) == 2.0

    # Test that the exception is handled when one is raised
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(4, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        return x / y

    # Test that the custom handler is called with the correct arguments
    assert func_with_custom_handler(4, 2) == 2.0
    assert len(handler_calls) == 0

    func_with_custom_handler(4, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Test that the generator works normally when no exception is raised
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8

    # Test that the exception is handled when one is raised in the generator
    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator function
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_func_with_custom_handler(x, y):
        yield x / y
        yield x * y

    gen = generator_func_with_custom_handler(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8
    assert len(handler_calls) == 0

    gen = generator_func_with_custom_handler(4, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 4
    assert y == 0
    assert kwargs == {}


# LLM-generated content at query #101
#--------------------------

```python
def test_log_exception():
    # Test basic exception logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

    # Test exception logging with user message
    try:
        raise TypeError("Type error")
    except TypeError as e:
        log_exception(e, "Custom user message")

    # Test subprocess.CalledProcessError with output
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


