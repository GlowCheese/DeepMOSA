####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

    # Should call custom handler
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
    assert next(gen) == 2
    assert next(gen) == 8

    # Should handle exception in generator
    gen = generator_func(1, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler_with_varargs(e, *args):
            pass

        @exception_wrapper(handler_with_varargs)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler_with_matching_default(e, x=1):
            pass

        @exception_wrapper(handler_with_matching_default)
        def func(x):
            pass


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None

    # Test that the hook is called with a non-skip exception
    def test_excepthook(type, value, traceback):
        test_excepthook.called = True
        test_excepthook.type = type
        test_excepthook.value = value
        test_excepthook.traceback = traceback

    test_excepthook.called = False
    original_excepthook = sys.excepthook
    sys.excepthook = test_excepthook

    try:
        raise ValueError("test")
    except ValueError:
        exc_info = sys.exc_info()
        sys.excepthook(*exc_info)

    assert test_excepthook.called
    assert test_excepthook.type is ValueError
    assert str(test_excepthook.value) == "test"
    assert test_excepthook.traceback is not None

    # Test that the hook is not called with a skip exception (KeyboardInterrupt)
    test_excepthook.called = False
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_info = sys.exc_info()
        sys.excepthook(*exc_info)

    assert not test_excepthook.called

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
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
        func_with_custom_handler(1, 0, z=5, extra="value")

    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 5, "kwargs": {"extra": "value"}}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x
        yield x / y

    gen = generator_func(1, 0)
    assert next(gen) == 1
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
        def func(x):
            pass

    # Test with handler argument having default value
    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, x=1: None)
        def func(x):
            pass


# LLM-generated content at query #4
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
        log_exception(e, extra_param="extra_value")


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y=2):
        return x / y

    # Should not raise
    assert func_with_default_handler(4, 2) == 2

    # Should log exception
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y=2, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2, **kwargs):
        return x / y

    # Should not raise
    assert func_with_custom_handler(4, 2) == 2

    # Should call custom handler
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0)
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ZeroDivisionError)
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 0
    assert handler_calls[0][3] == {}

    # Test with generator
    @exception_wrapper()
    def generator_func(x):
        yield x
        raise ValueError("test")

    gen = generator_func(1)
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

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
        exception_wrapper(bad_handler3)(lambda x: None)

    def bad_handler4(e, x=1):
        pass

    with pytest.raises(ValueError, match="Argument 'x' matches wrapped method argument"):
        exception_wrapper(bad_handler4)(lambda x: None)


# LLM-generated content at query #6
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

    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, my_arg, kw)

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, my_arg=None, **kwargs):
        raise TypeError("Custom handler test")

    func_with_custom_handler(1, 2, my_arg=3, extra=4)
    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == 3
    assert handler_args[4] == {"kwargs": {"extra": 4}}

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

    # Test with custom handler and generator
    handler_called = False

    @exception_wrapper(custom_handler)
    def generator_with_custom_handler(arg1):
        yield arg1
        raise StopIteration("Generator custom handler test")

    gen = generator_with_custom_handler(10)
    assert next(gen) == 10
    with pytest.raises(StopIteration):
        next(gen)
    assert handler_called

    # Test successful execution (no exception)
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

    # Test with generator that doesn't raise
    @exception_wrapper()
    def generator_no_error():
        yield 1
        yield 2

    gen = generator_no_error()
    assert list(gen) == [1, 2]

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=None: None)
        def func_default_mismatch(arg1):
            pass

    # Test handler with no exception argument
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_no_exc_arg():
            pass


# LLM-generated content at query #7
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    original_excepthook = sys.excepthook

    # Verify excepthook is set
    assert sys.excepthook is not None

    # Test with KeyboardInterrupt (should not trigger IPython)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    # Test with other exception (should trigger IPython)
    try:
        raise ValueError("test")
    except ValueError:
        # Mock the IPython hook to verify it's called
        import IPython.core.ultratb as ultratb
        original_call_pdb = ultratb.FormattedTB.call_pdb
        ultratb.FormattedTB.call_pdb = lambda self, *args: None

        try:
            sys.excepthook(ValueError, ValueError("test"), None)
        finally:
            ultratb.FormattedTB.call_pdb = original_call_pdb

    # Test with capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    # Verify KeyboardInterrupt now triggers IPython
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        # Mock the IPython hook to verify it's called
        ultratb.FormattedTB.call_pdb = lambda self, *args: None
        try:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        finally:
            ultratb.FormattedTB.call_pdb = original_call_pdb

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #8
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

    # Test logging exception when another exception occurs during logging
    class MockException(Exception):
        pass

    original_log = log
    def mock_log(*args, **kwargs):
        raise MockException("Logging failed")

    log = mock_log
    try:
        try:
            raise RuntimeError("Original error")
        except RuntimeError as e:
            with pytest.raises(MockException):
                log_exception(e)
    finally:
        log = original_log


# LLM-generated content at query #9
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
        raise TypeError("Another test error")
    except TypeError as e:
        log_exception(e, "Custom error message")

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "test_command", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Test with kwargs")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


# LLM-generated content at query #10
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

    # Test successful execution
    @exception_wrapper()
    def successful_func(x, y):
        return x + y

    assert successful_func(1, 2) == 3

    # Test with custom handler and successful execution
    handler_calls.clear()
    assert successful_func(1, 2) == 3
    assert len(handler_calls) == 0

    # Test handler with mismatched arguments (should raise ValueError)
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_handler)(lambda: None)

    # Test handler with default values on matched arguments (should raise ValueError)
    def another_bad_handler(e, x=1):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(another_bad_handler)(lambda x: None)


# LLM-generated content at query #11
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
        type_, value, tb = sys.exc_info()
        sys.excepthook(type_, value, tb)
        # Should not enter IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        type_, value, tb = sys.exc_info()
        # Mock the IPython hook to avoid actually launching IPython
        import IPython.core.ultratb as ultratb
        original_call_pdb = ultratb.FormattedTB.call_pdb
        ultratb.FormattedTB.call_pdb = lambda self, etype, value, tb: None
        sys.excepthook(type_, value, tb)
        ultratb.FormattedTB.call_pdb = original_call_pdb

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        type_, value, tb = sys.exc_info()
        # Mock the IPython hook to avoid actually launching IPython
        import IPython.core.ultratb as ultratb
        original_call_pdb = ultratb.FormattedTB.call_pdb
        ultratb.FormattedTB.call_pdb = lambda self, etype, value, tb: None
        sys.excepthook(type_, value, tb)
        ultratb.FormattedTB.call_pdb = original_call_pdb

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise exception
    assert func_with_default_handler(10, 2) == 5

    # Should handle exception
    func_with_default_handler(10, 0)

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

    # Should not raise exception
    assert func_with_custom_handler(10, 2) == 5

    # Should call custom handler
    func_with_custom_handler(10, 0, z=5)
    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 10
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 5}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    # Should handle exception in generator
    gen = generator_func(10, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test with invalid handler (no exception argument)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_with_invalid_handler():
            pass

    # Test with invalid handler (varargs)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func_with_varargs_handler():
            pass

    # Test with invalid handler (mismatched args)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, z: None)
        def func_with_mismatched_handler(x, y):
            pass

    # Test with invalid handler (default values on matched args)
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_handler(x):
            pass


# LLM-generated content at query #13
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
        pass  # The exception should not be captured

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass  # The exception should be captured

    # Restore the original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")

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
        raise RuntimeError("custom handler test")

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
        raise StopIteration("generator test")

    gen = gen_func()
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


# LLM-generated content at query #15
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

    # Test with subprocess.CalledProcessError and output
    try:
        raise subprocess.CalledProcessError(1, "test_cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "test_cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Test runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


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

    # Should not raise
    assert func_with_custom_handler(4, 2) == 2

    # Should call custom handler
    func_with_custom_handler(1, 0)
    assert handler_called
    assert isinstance(handler_args[0], ZeroDivisionError)
    assert handler_args[1] == 1
    assert handler_args[2] == 0
    assert handler_args[3] == {'z': 10}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x):
        yield x
        yield 1 / 0  # This should trigger the exception

    gen = generator_func(1)
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass  # Exception was handled

    # Test with varargs in handler (should raise ValueError)
    def bad_handler_with_varargs(e, *args):
        pass

    try:
        exception_wrapper(bad_handler_with_varargs)(lambda: None)
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "varargs" in str(ve)

    # Test with no exception argument in handler (should raise ValueError)
    def bad_handler_no_exception():
        pass

    try:
        exception_wrapper(bad_handler_no_exception)(lambda: None)
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "positional argument" in str(ve)

    # Test with handler argument not matching wrapped function (should raise ValueError)
    def bad_handler_mismatch(e, nonexistent_arg):
        pass

    try:
        exception_wrapper(bad_handler_mismatch)(lambda x: x)
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "does not match" in str(ve)

    # Test with handler argument matching wrapped function but having default (should raise ValueError)
    def bad_handler_with_default(e, x=1):
        pass

    try:
        exception_wrapper(bad_handler_with_default)(lambda x: x)
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "cannot have default values" in str(ve)


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.excepthook(type, value, traceback)
    assert True  # If we get here, the exception was not captured

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        sys.excepthook(type, value, traceback)
    assert True  # If we get here, the exception was captured

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit
    except BdbQuit:
        sys.excepthook(type, value, traceback)
    assert True  # If we get here, the exception was not captured

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #18
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

    gen = generator_func(4, 2)
    assert next(gen) == 2
    assert next(gen) == 8

    gen = generator_func(4, 0)
    with pytest.raises(ZeroDivisionError):
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

    # Test handler with non-matching argument (should raise ValueError)
    def yet_another_bad_handler(e, non_existent_arg):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(yet_another_bad_handler)

    # Test handler with default value for matching argument (should raise ValueError)
    def bad_default_handler(e, x=10):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(bad_default_handler)


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper():
    # Test default handler (log_exception)
    @exception_wrapper()
    def func_raises_value_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        func_raises_value_error()

    # Test custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Test runtime error")

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
        raise StopIteration("Test stop")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatch():
            pass

    # Test handler with default values on matching args
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_with_default(arg1):
            pass


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Test that exception is logged
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

    # Test that custom handler is called with correct arguments
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0, z=5)
    assert handler_called
    assert isinstance(handler_args["e"], ZeroDivisionError)
    assert handler_args["x"] == 1
    assert handler_args["y"] == 0
    assert handler_args["kwargs"] == {"z": 5}

    # Test that handler with default values works
    def handler_with_defaults(e, x, default_arg="default", **kwargs):
        return {"e": e, "x": x, "default_arg": default_arg, "kwargs": kwargs}

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        return x / y

    result = None
    try:
        func_with_defaults(1, 0)
    except ZeroDivisionError:
        pass
    assert result is None  # Handler should not affect the exception propagation

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y

    # Test that exception in generator is caught
    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test that handler arguments match function arguments
    def mismatched_handler(e, non_existent_arg, **kwargs):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent_arg' in exception handler does not match"):
        exception_wrapper(mismatched_handler)(lambda: None)

    # Test that handler cannot have varargs
    def handler_with_varargs(e, *args, **kwargs):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(handler_with_varargs)(lambda: None)

    # Test that handler must have exception argument
    def handler_without_exception():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument for the exception object"):
        exception_wrapper(handler_without_exception)(lambda: None)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y):
        return x / y

    # Should not raise exception
    assert func_with_default_handler(10, 2) == 5

    # Should handle exception and not raise
    func_with_default_handler(10, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=3):
        return x / y

    # Should not raise exception
    assert func_with_custom_handler(10, 2) == 5

    # Should call custom handler
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

    # Should not raise exception
    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    # Should handle exception in generator
    gen = generator_func(10, 0)
    with pytest.raises(StopIteration):
        next(gen)

    # Test with invalid handler
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func_with_invalid_handler():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda *args: None)
        def func_with_varargs_handler():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_with_mismatched_handler():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x=1: None)
        def func_with_default_handler(x):
            pass


# LLM-generated content at query #2
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

    # Test successful execution
    assert func_with_default_handler(10, 2) == 5
    assert func_with_custom_handler(10, 2, z=5) == 5

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y):
        yield x / y
        yield x * y

    gen = generator_func(10, 2)
    assert next(gen) == 5
    assert next(gen) == 20

    gen_with_error = generator_func(10, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen_with_error)

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def bad_func(x):
            return x

    # Test handler with default values on matched arguments
    def another_bad_handler(e, x=5):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(another_bad_handler)
        def another_bad_func(x):
            return x

    # Test handler with varargs
    def varargs_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(varargs_handler)
        def varargs_func(x):
            return x


# LLM-generated content at query #3
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        type, value, tb = sys.exc_info()
        sys.excepthook(type, value, tb)
        assert True  # If we get here, the exception was not captured

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        type, value, tb = sys.exc_info()
        # This should not raise an exception
        sys.excepthook(type, value, tb)

    # Reset the excepthook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        type, value, tb = sys.exc_info()
        # This should not raise an exception
        sys.excepthook(type, value, tb)

    # Reset the excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #4
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
    error = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    log_exception(error)

    # Test subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")

    # Test exception during logging
    class BadException(Exception):
        def __str__(self):
            raise RuntimeError("Bad __str__")

    try:
        raise BadException()
    except BadException as e:
        with pytest.raises(RuntimeError):
            log_exception(e)


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct excepthook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is skipped by default
    skip_exceptions = [BdbQuit, KeyboardInterrupt]
    for exc_type in skip_exceptions:
        try:
            raise exc_type
        except:
            sys.excepthook(exc_type, exc_type(), None)
            assert True  # Should not reach here if exception is not skipped

    # Test that other exceptions are handled by IPython
    try:
        raise ValueError("test")
    except:
        sys.excepthook(ValueError, ValueError("test"), None)
        assert True  # Should reach here if exception is handled

    # Test that KeyboardInterrupt is not skipped when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        assert True  # Should reach here if exception is handled


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_with_default_handler(x, y=2):
        return x / y

    # Test normal execution
    assert func_with_default_handler(4, 2) == 2.0

    # Test exception handling
    with pytest.raises(ZeroDivisionError):
        func_with_default_handler(1, 0)

    # Test with custom handler
    handler_calls = []

    def custom_handler(e, x, y=2, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y=2, **kwargs):
        return x / y

    # Test normal execution
    assert func_with_custom_handler(4, 2) == 2.0
    assert len(handler_calls) == 0

    # Test exception handling
    with pytest.raises(ZeroDivisionError):
        func_with_custom_handler(1, 0)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {}

    # Test with generator function
    @exception_wrapper()
    def generator_func(x, y=2):
        yield x / y
        yield x * y

    # Test normal execution
    gen = generator_func(4, 2)
    assert next(gen) == 2.0
    assert next(gen) == 8
    with pytest.raises(StopIteration):
        next(gen)

    # Test exception handling in generator
    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)

    # Test with custom handler and generator
    handler_calls.clear()

    @exception_wrapper(custom_handler)
    def generator_func_custom(x, y=2):
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

    # Test handler with varargs (should raise ValueError)
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        def bad_handler(e, *args):
            pass

        @exception_wrapper(bad_handler)
        def bad_func():
            pass

    # Test handler with no args (should raise ValueError)
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        def no_arg_handler():
            pass

        @exception_wrapper(no_arg_handler)
        def no_arg_func():
            pass

    # Test handler with mismatched args (should raise ValueError)
    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        def mismatched_handler(e, z):
            pass

        @exception_wrapper(mismatched_handler)
        def mismatched_func(x, y=2):
            pass

    # Test handler with default args that match wrapped method (should raise ValueError)
    with pytest.raises(ValueError, match="cannot have default values"):
        def default_arg_handler(e, x=1):
            pass

        @exception_wrapper(default_arg_handler)
        def default_arg_func(x, y=2):
            pass


# LLM-generated content at query #8
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
    def func_with_custom_handler(x, y, z=3, **kwargs):
        return x / y

    func_with_custom_handler(1, 0, z=5, extra=10)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 1
    assert y == 0
    assert kwargs == {"z": 5, "extra": 10}

    # Test custom handler with generator
    @exception_wrapper(custom_handler)
    def generator_func(x, y):
        yield x / y

    gen = generator_func(1, 0)
    with pytest.raises(ZeroDivisionError):
        next(gen)
    assert len(handler_calls) == 2  # One for the generator creation, one for the exception

    # Test successful execution
    @exception_wrapper(custom_handler)
    def successful_func(x, y):
        return x + y

    result = successful_func(1, 2)
    assert result == 3
    assert len(handler_calls) == 2  # No new calls

    # Test handler with mismatched arguments
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def func_with_bad_handler(x, y):
            return x / y

    # Test handler with default values on matching args
    def handler_with_defaults(e, x=1, y=2):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func_with_defaults_handler(x, y):
            return x / y


# LLM-generated content at query #9
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

    # Test with subprocess.CalledProcessError and output
    try:
        raise subprocess.CalledProcessError(1, "test_command", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with subprocess.CalledProcessError without output
    try:
        raise subprocess.CalledProcessError(1, "test_command")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, extra_kwarg="extra_value")


# LLM-generated content at query #10
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
    error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    log_exception(error)

    # Test with subprocess.CalledProcessError without output
    error = subprocess.CalledProcessError(1, "cmd")
    log_exception(error)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")


# LLM-generated content at query #11
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered correctly
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    def test_excepthook(type, value, traceback):
        assert type is not KeyboardInterrupt

    sys.excepthook = test_excepthook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    sys.excepthook = test_excepthook
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #12
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
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    func_with_args(1, 2, extra_kw="extra_value")

    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == 1
    assert handler_args['arg2'] == 2
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {'extra_kw': 'extra_value'}

    # Test with generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator error")

    gen = generator_func()
    with pytest.raises(StopIteration):
        next(gen)
        next(gen)

    # Test handler validation
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler_with_varargs(e, *args):
            pass

        @exception_wrapper(handler_with_varargs)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler_with_mismatched_args(e, nonexistent_arg):
            pass

        @exception_wrapper(handler_with_mismatched_args)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler_with_default_on_matched_arg(e, arg1="default"):
            pass

        @exception_wrapper(handler_with_default_on_matched_arg)
        def func(arg1):
            pass


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test default behavior (KeyboardInterrupt not captured)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None

    # Test with KeyboardInterrupt captured
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None

    # Test that the excepthook is called with the correct arguments
    def mock_excepthook(type, value, traceback):
        mock_excepthook.called = True
        mock_excepthook.args = (type, value, traceback)

    mock_excepthook.called = False
    mock_excepthook.args = None

    # Replace the IPython hook with our mock
    original_excepthook = sys.excepthook
    try:
        # Simulate an exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()
            sys.excepthook(*exc_info)

        assert mock_excepthook.called
        assert mock_excepthook.args[0] is ValueError
        assert isinstance(mock_excepthook.args[1], ValueError)
        assert mock_excepthook.args[2] is not None
    finally:
        sys.excepthook = original_excepthook


# LLM-generated content at query #14
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the excepthook is registered
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Test that BdbQuit is not captured
    try:
        raise BdbQuit()
    except BdbQuit:
        pass

    # Test that other exceptions are captured
    try:
        raise ValueError("test")
    except ValueError:
        pass

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #15
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test that the function registers the correct exception hook
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)

    # Test that KeyboardInterrupt is not captured by default
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        assert True  # Should not enter IPython debugger

    # Test that other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass  # Should not raise because the exception hook is called

    # Reset the exception hook
    sys.excepthook = original_excepthook

    # Test that KeyboardInterrupt is captured when capture_keyboard_interrupt is True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass  # Should not raise because the exception hook is called

    # Reset the exception hook
    sys.excepthook = original_excepthook


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")

    # Test that the function raises no exception (handled internally)
    func_raises_error()

    # Test with custom handler
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, default_arg, kwargs)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("Custom handler test")

    # Test custom handler is called with correct arguments
    func_with_args("val1", "val2", extra_kwarg="extra")
    assert handler_called
    e, arg1, arg2, default_arg, kwargs = handler_args
    assert isinstance(e, RuntimeError)
    assert str(e) == "Custom handler test"
    assert arg1 == "val1"
    assert arg2 == "val2"
    assert default_arg == "default"
    assert kwargs == {"extra_kwarg": "extra"}

    # Test generator function
    @exception_wrapper()
    def generator_func():
        yield 1
        raise StopIteration("Generator test")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test that handler with wrong signature raises ValueError
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, non_matching_arg: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, matching_arg="default": None)
        def func(matching_arg):
            pass


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def test_func_default():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func_default()

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
    def test_func_custom(arg1, arg2, default_arg="default", **kwargs):
        raise TypeError("Custom test error")

    with pytest.raises(TypeError):
        test_func_custom("val1", "val2", extra_kw="extra")

    assert handler_called
    assert isinstance(handler_args['e'], TypeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kw": "extra"}

    # Test with generator function
    @exception_wrapper()
    def test_generator():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

    # Test handler validation
    def bad_handler1():
        pass

    def bad_handler2(e, *args):
        pass

    def bad_handler3(e, arg1="default"):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(bad_handler1)(lambda: None)

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(bad_handler2)(lambda: None)

    with pytest.raises(ValueError, match="Argument 'arg1' in exception handler does not match"):
        exception_wrapper(bad_handler3)(lambda: None)


# LLM-generated content at query #18
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
        log_exception(e, extra_arg="extra_value")


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper():
    # Test with default handler (log_exception)
    @exception_wrapper()
    def test_func_default(x, y):
        return x / y

    # Test with custom handler
    handler_calls = []
    def custom_handler(e, x, y, **kwargs):
        handler_calls.append((e, x, y, kwargs))

    @exception_wrapper(custom_handler)
    def test_func_custom(x, y, z=10):
        return x / y

    # Test with generator function
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Generator error")
        yield 2

    # Test default handler
    with pytest.raises(ZeroDivisionError):
        test_func_default(1, 0)

    # Test custom handler
    test_func_custom(5, 0, z=20)
    assert len(handler_calls) == 1
    e, x, y, kwargs = handler_calls[0]
    assert isinstance(e, ZeroDivisionError)
    assert x == 5
    assert y == 0
    assert kwargs == {'z': 20}

    # Test generator
    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

    # Test successful execution
    assert test_func_default(10, 2) == 5
    assert test_func_custom(10, 2) == 5

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
        raise subprocess.CalledProcessError(1, "cmd", output=b"error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="extra_value")


# LLM-generated content at query #21
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
    else:
        assert False, "KeyboardInterrupt should not be captured by default"

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should be captured when specified"

    # Test that other exceptions are captured
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass
    else:
        assert False, "ValueError should be captured"


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper():
    # Test basic exception handling with default handler
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_raises()

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
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("custom handler test")

    func_with_args("val1", "val2", extra_kw="extra")
    assert handler_called
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['arg1'] == "val1"
    assert handler_args['arg2'] == "val2"
    assert handler_args['default_arg'] == "default"
    assert handler_args['kwargs'] == {"extra_kw": "extra"}

    # Test generator function
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator test")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    # Test handler with mismatched arguments
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func_mismatch():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def func_default(arg1):
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


# LLM-generated content at query #23
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

    # Test subprocess.CalledProcessError logging
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Runtime error")
    except RuntimeError as e:
        log_exception(e, "Runtime error occurred", level="critical", extra_info="additional data")


# LLM-generated content at query #24
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
    assert True

    # Test that KeyboardInterrupt is captured when specified
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert True

    # Test that BdbQuit is not captured
    register_ipython_excepthook()
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    assert True

    # Test that other exceptions are captured
    register_ipython_excepthook()
    try:
        raise ValueError("test")
    except ValueError:
        pass
    assert True


# LLM-generated content at query #25
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
        raise subprocess.CalledProcessError(1, "cmd", output=b"Error output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

    # Test exception logging with additional kwargs
    try:
        raise RuntimeError("Test with kwargs")
    except RuntimeError as e:
        log_exception(e, "User message", extra_kwarg="value")


