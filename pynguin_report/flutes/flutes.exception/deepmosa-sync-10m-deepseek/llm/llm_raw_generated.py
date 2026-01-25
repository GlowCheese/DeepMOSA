####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #2
#--------------------------

def test_log_exception_with_user_msg():
    user_msg = "Custom error message"
    exception = ValueError("Test exception")
    log_exception(exception, user_msg)

def test_log_exception_without_user_msg():
    exception = RuntimeError("Another test exception")
    log_exception(exception)

def test_log_exception_called_process_error_with_output():
    exception = subprocess.CalledProcessError(1, "cmd", output=b"output")
    log_exception(exception)

def test_log_exception_called_process_error_without_output():
    exception = subprocess.CalledProcessError(1, "cmd", output=None)
    log_exception(exception)

def test_log_exception_logging_failure():
    original_log = log
    log = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Logging failed"))
    exception = Exception("Original exception")
    try:
        log_exception(exception)
    except Exception as e:
        assert str(e) == "Logging failed"
    log = original_log

def test_log_exception_with_additional_kwargs():
    exception = TypeError("Type error")
    log_exception(exception, force_console=True, timestamp=False)


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_called_process_error_with_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command '['ls']' returned non-zero exit status 1.", "error")
        assert not mock_log.call_args_list[0][0][0].startswith("Traceback")


# LLM-generated content at query #4
#--------------------------

def test_log_exception_with_user_msg():
    user_msg = "Custom error"
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, user_msg=user_msg, force_console=False)

def test_log_exception_without_user_msg():
    try:
        raise RuntimeError("Runtime failure")
    except RuntimeError as e:
        log_exception(e, force_console=False)

def test_log_exception_with_called_process_error():
    class MockCalledProcessError(Exception):
        output = "Command output"
    try:
        raise MockCalledProcessError()
    except MockCalledProcessError as e:
        log_exception(e, force_console=False)

def test_log_exception_logging_failure():
    original_log = log
    log_called = False
    def mock_log(*args, **kwargs):
        nonlocal log_called
        log_called = True
        raise RuntimeError("Logging failed")
    import flutes.exception
    flutes.exception.log = mock_log
    try:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, force_console=False)
    except RuntimeError as re:
        assert str(re) == "Logging failed"
    finally:
        flutes.exception.log = original_log
    assert log_called == True

def test_log_exception_with_additional_kwargs():
    try:
        raise TypeError("Type mismatch")
    except TypeError as e:
        log_exception(e, timestamp=False, include_proc_id=False, force_console=False)


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_with_matching_arguments():
    handler_args = {}
    def custom_handler(e, arg1, arg2):
        handler_args['e'] = e
        handler_args['arg1'] = arg1
        handler_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def func_that_raises(arg1, arg2):
        raise ValueError("test error")
    func_that_raises("value1", "value2")
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['arg2'] == "value2"

def test_exception_wrapper_with_kwargs():
    handler_args = {}
    def custom_handler(e, arg1, **kwargs):
        handler_args['e'] = e
        handler_args['arg1'] = arg1
        handler_args['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func_that_raises(arg1, arg2=None, **kwargs):
        raise ValueError("test error")
    func_that_raises("value1", arg2="value2", extra="extra")
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['kwargs'] == {'arg2': 'value2', 'kwargs': {'extra': 'extra'}}

def test_exception_wrapper_with_default_values_in_handler():
    handler_args = {}
    def custom_handler(e, arg1, optional_arg="default"):
        handler_args['e'] = e
        handler_args['arg1'] = arg1
        handler_args['optional_arg'] = optional_arg
    @exception_wrapper(custom_handler)
    def func_that_raises(arg1):
        raise ValueError("test error")
    func_that_raises("value1")
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['arg1'] == "value1"
    assert handler_args['optional_arg'] == "default"

def test_exception_wrapper_with_generator():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def generator_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = generator_func()
    result = list(gen)
    assert result == [1]
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_with_nested_wrapped():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    @exception_wrapper(custom_handler)
    @decorator
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)

def test_exception_wrapper_with_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_mismatched_argument():
    try:
        def handler(e, non_existent_arg):
            pass
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_matching_argument_with_default():
    try:
        def handler(e, arg_with_default="default"):
            pass
        @exception_wrapper(handler)
        def func(arg_with_default):
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_successful_execution():
    @exception_wrapper()
    def normal_func(x, y):
        return x + y
    result = normal_func(2, 3)
    assert result == 5

def test_exception_wrapper_generator_successful_execution():
    @exception_wrapper()
    def normal_generator(n):
        for i in range(n):
            yield i
    result = list(normal_generator(3))
    assert result == [0, 1, 2]


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_fn():
        pass
    try:
        exception_wrapper(handler_fn)(lambda: None)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #7
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def failing_function():
        raise ValueError("test error")
    failing_function()

def test_exception_wrapper_passes_through_return_value():
    @exception_wrapper()
    def successful_function():
        return 42
    result = successful_function()
    assert result == 42

def test_exception_wrapper_wraps_generator():
    @exception_wrapper()
    def generator_function():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = generator_function()
    assert list(gen) == [1]

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def failing_function():
        raise RuntimeError("custom error")
    failing_function()
    assert isinstance(caught_exception, RuntimeError)
    assert str(caught_exception) == "custom error"

def test_exception_wrapper_handler_with_matching_args():
    handler_args = {}
    def custom_handler(e, arg1, arg2):
        handler_args['e'] = e
        handler_args['arg1'] = arg1
        handler_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("match error")
    failing_function(10, arg2=20)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['arg1'] == 10
    assert handler_args['arg2'] == 20

def test_exception_wrapper_handler_with_kwargs():
    handler_kwargs = {}
    def custom_handler(e, **kwargs):
        handler_kwargs['e'] = e
        handler_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def failing_function(a, b, c=30):
        raise TypeError("kwargs error")
    failing_function(1, 2)
    assert isinstance(handler_kwargs['e'], TypeError)
    assert handler_kwargs['a'] == 1
    assert handler_kwargs['b'] == 2
    assert handler_kwargs['c'] == 30

def test_exception_wrapper_handler_with_default_args():
    handler_values = {}
    def custom_handler(e, required, optional=100):
        handler_values['e'] = e
        handler_values['required'] = required
        handler_values['optional'] = optional
    @exception_wrapper(custom_handler)
    def failing_function(required):
        raise Exception("default error")
    failing_function(999)
    assert isinstance(handler_values['e'], Exception)
    assert handler_values['required'] == 999
    assert handler_values['optional'] == 100

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_missing_handler_arg():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def dummy():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_handler_arg_with_default_matches():
    try:
        def handler(e, arg=5):
            pass
        @exception_wrapper(handler)
        def dummy(arg):
            pass
        assert False
    except ValueError:
        pass


# LLM-generated content at query #8
#--------------------------

def test_log_exception_with_user_msg():
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="User message")
    assert True

def test_log_exception_without_user_msg():
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
    assert True

def test_log_exception_with_called_process_error():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    assert True

def test_log_exception_with_additional_kwargs():
    try:
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)
    assert True

def test_log_exception_logging_failure():
    original_log = log
    log_called = False
    def mock_log(*args, **kwargs):
        nonlocal log_called
        log_called = True
        raise RuntimeError("log failure")
    log = mock_log
    try:
        raise ValueError("error")
    except ValueError as e:
        try:
            log_exception(e)
        except RuntimeError as log_e:
            assert str(log_e) == "log failure"
    finally:
        log = original_log
    assert log_called


# LLM-generated content at query #9
#--------------------------

def test_log_exception_with_user_message():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="User message")
    assert True

def test_log_exception_without_user_message():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
    assert True

def test_log_exception_with_called_process_error():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    assert True

def test_log_exception_with_logging_error():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    original_log = log
    def failing_log(*args, **kwargs):
        raise RuntimeError("logging failed")
    log = failing_log
    try:
        raise ValueError("test error")
    except ValueError as e:
        try:
            log_exception(e)
        except RuntimeError as log_e:
            assert str(log_e) == "logging failed"
    log = original_log

def test_log_exception_with_additional_kwargs():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise KeyError("missing key")
    except KeyError as e:
        log_exception(e, force_console=True, timestamp=False)
    assert True


# LLM-generated content at query #10
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    with patch("flutes.log.log") as mock_log:
        log_exception(e)
        mock_log.assert_called_once()
        call_args = mock_log.call_args_list
        assert len(call_args) == 1
        assert call_args[0][0][1] == "error"


# LLM-generated content at query #11
#--------------------------

def test_log_exception_with_user_message():
    user_msg = "Custom error"
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, user_msg=user_msg)

def test_log_exception_without_user_message():
    try:
        raise RuntimeError("Runtime failure")
    except RuntimeError as e:
        log_exception(e)

def test_log_exception_with_kwargs():
    try:
        raise TypeError("Type mismatch")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)

def test_log_exception_called_process_error_with_output():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

def test_log_exception_called_process_error_without_output():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=None)
    except subprocess.CalledProcessError as e:
        log_exception(e)

def test_log_exception_logging_failure():
    original_log = log
    log_called = False
    def mock_log(*args, **kwargs):
        nonlocal log_called
        log_called = True
        raise RuntimeError("Logging failed")
    log = mock_log
    try:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e)
    except RuntimeError as e:
        assert str(e) == "Logging failed"
    finally:
        log = original_log
    assert log_called


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_error():
    def handler_with_varargs(e, *args):
        pass
    decorator = exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_value_error():
    def handler_with_varargs(e, *args):
        pass

    try:
        exception_wrapper(handler_with_varargs)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_value_error():
    def handler_with_varargs(e, *args):
        pass
    decorator = exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #16
#--------------------------

def test_log_exception_with_non_called_process_error():
    e = ValueError("test error")
    log_exception(e)
    assert True

def test_log_exception_with_called_process_error_no_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test")
    e.output = None
    log_exception(e)
    assert True

def test_log_exception_with_called_process_error_with_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test")
    e.output = "output"
    log_exception(e)
    assert True


# LLM-generated content at query #17
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler_with_matching_args():
    captured_exception = None
    captured_arg = None
    def handler(e, arg):
        nonlocal captured_exception, captured_arg
        captured_exception = e
        captured_arg = arg
    @exception_wrapper(handler)
    def func(arg):
        raise RuntimeError("custom error")
    func("test_arg")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_arg == "test_arg"

def test_exception_wrapper_custom_handler_with_default_args():
    captured_exception = None
    captured_arg = None
    captured_default = None
    def handler(e, arg, default="default_value"):
        nonlocal captured_exception, captured_arg, captured_default
        captured_exception = e
        captured_arg = arg
        captured_default = default
    @exception_wrapper(handler)
    def func(arg):
        raise KeyError("key error")
    func("test_arg")
    assert isinstance(captured_exception, KeyError)
    assert captured_arg == "test_arg"
    assert captured_default == "default_value"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_exception = None
    captured_kwargs = None
    def handler(e, **kwargs):
        nonlocal captured_exception, captured_kwargs
        captured_exception = e
        captured_kwargs = kwargs
    @exception_wrapper(handler)
    def func(a, b, c=3):
        raise IndexError("index error")
    func(1, b=2)
    assert isinstance(captured_exception, IndexError)
    assert captured_kwargs == {'a': 1, 'b': 2, 'c': 3}

def test_exception_wrapper_custom_handler_with_mixed_args():
    captured_exception = None
    captured_a = None
    captured_b = None
    captured_kwargs = None
    def handler(e, a, b, extra="extra", **kwargs):
        nonlocal captured_exception, captured_a, captured_b, captured_kwargs
        captured_exception = e
        captured_a = a
        captured_b = b
        captured_kwargs = kwargs
    @exception_wrapper(handler)
    def func(a, b, c=3, d=4):
        raise TypeError("type error")
    func(10, b=20, d=40)
    assert isinstance(captured_exception, TypeError)
    assert captured_a == 10
    assert captured_b == 20
    assert captured_kwargs == {'c': 3, 'd': 40}

def test_exception_wrapper_generator_function():
    error_occurred = False
    def handler(e):
        nonlocal error_occurred
        error_occurred = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    list(gen)
    assert error_occurred

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    gen = gen_func()
    assert list(gen) == [0, 1, 2]

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "Exception handler must have a positional argument for the exception object" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument (*args)" in str(e)

def test_exception_wrapper_handler_arg_not_in_wrapped():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "Argument 'missing_arg' in exception handler does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches_wrapped():
    try:
        def handler(e, arg="default"):
            pass
        @exception_wrapper(handler)
        def func(arg):
            pass
        func("test")
    except ValueError as e:
        assert "Argument 'arg' matches wrapped method argument, thus cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function_with_args_kwargs():
    captured_exception = None
    captured_args = None
    captured_kwargs = None
    def handler(e, *args, **kwargs):
        nonlocal captured_exception, captured_args, captured_kwargs
        captured_exception = e
        captured_args = args
        captured_kwargs = kwargs
    @exception_wrapper(handler)
    def func(*args, **kwargs):
        raise AssertionError("assertion error")
    func(1, 2, a=3, b=4)
    assert isinstance(captured_exception, AssertionError)
    assert captured_args == (1, 2)
    assert captured_kwargs == {'a': 3, 'b': 4}

def test_exception_wrapper_handler_with_kwonly_args():
    captured_exception = None
    captured_kwonly = None
    def handler(e, *, kwonly):
        nonlocal captured_exception, captured_kwonly
        captured_exception = e
        captured_kwonly = kwonly
    @exception_wrapper(handler)
    def func(*, kwonly):
        raise OSError("os error")
    func(kwonly="value")
    assert isinstance(captured_exception, OSError)
    assert captured_kwonly == "value"

def test_exception_wrapper_handler_with_positional_only():
    captured_exception = None
    captured_pos = None
    def handler(e, pos):
        nonlocal captured_exception, captured_pos
        captured_exception = e
        captured_pos = pos
    @exception_wrapper(handler)
    def func(pos, /):
        raise MemoryError("memory error")
    func(42)
    assert isinstance(captured_exception, MemoryError)
    assert captured_pos == 42


# LLM-generated content at query #18
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_skip_exceptions():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    hook = sys.excepthook
    sys.excepthook = original_hook
    assert hook is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)(lambda: None)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception():
        pass
    try:
        @exception_wrapper(handler_without_exception)
        def foo():
            pass
        foo()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #21
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #22
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured = {}
    def custom_handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, arg2=20)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == 20

def test_exception_wrapper_custom_handler_with_default_args():
    captured = {}
    def custom_handler(e, arg1, arg2, extra="default"):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['extra'] = extra
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, arg2=15)
    assert captured['arg1'] == 5
    assert captured['arg2'] == 15
    assert captured['extra'] == "default"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def custom_handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, arg3=30):
        raise RuntimeError("error")
    func(1, arg2=2)
    assert captured['arg1'] == 1
    assert captured['kwargs'] == {'arg2': 2, 'arg3': 30}

def test_exception_wrapper_custom_handler_with_kwonlyargs():
    captured = {}
    def custom_handler(e, arg1, *, kwonly):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwonly'] = kwonly
    @exception_wrapper(custom_handler)
    def func(arg1, *, kwonly):
        raise RuntimeError("error")
    func(100, kwonly=200)
    assert captured['arg1'] == 100
    assert captured['kwonly'] == 200

def test_exception_wrapper_custom_handler_with_var_kw():
    captured = {}
    def custom_handler(e, arg1, **extra):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['extra'] = extra
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, arg3=300):
        raise RuntimeError("error")
    func(7, arg2=8)
    assert captured['arg1'] == 7
    assert captured['extra'] == {'arg2': 8, 'arg3': 300}

def test_exception_wrapper_custom_handler_with_args_and_kwargs():
    captured = {}
    def custom_handler(e, arg1, *args, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['args'] = args
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func(arg1, *args, arg2, **kwargs):
        raise RuntimeError("error")
    func(1, 2, 3, arg2=4, extra=5)
    assert captured['arg1'] == 1
    assert captured['args'] == (2, 3)
    assert captured['kwargs'] == {'arg2': 4, 'extra': 5}

def test_exception_wrapper_generator_function():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_generator_function_with_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    list(gen)
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_normal_return():
    @exception_wrapper()
    def normal_func():
        return 42
    result = normal_func()
    assert result == 42

def test_exception_wrapper_generator_return():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    result = list(gen_func())
    assert result == [0, 1, 2]

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def bad_handler():
            pass
        @exception_wrapper(bad_handler)
        def dummy():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument for the exception object" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def bad_handler(e, *args):
            pass
        @exception_wrapper(bad_handler)
        def dummy():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument (*args)" in str(e)

def test_exception_wrapper_handler_arg_not_in_wrapped():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def dummy(arg1):
            pass
    except ValueError as e:
        assert "Argument 'missing_arg' in exception handler does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches_wrapped():
    try:
        def handler(e, arg1, arg2="default"):
            pass
        @exception_wrapper(handler)
        def dummy(arg1, arg2):
            pass
    except ValueError as e:
        assert "Argument 'arg2' matches wrapped method argument, thus cannot have default values" in str(e)


# LLM-generated content at query #23
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_passes_arguments_to_handler():
    captured_args = {}
    def handler(e, arg1, arg2, kwarg1=None):
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
        captured_args['kwarg1'] = kwarg1
    @exception_wrapper(handler)
    def func(arg1, arg2, kwarg1=None):
        raise RuntimeError("error")
    func(1, "two", kwarg1=3.0)
    assert captured_args['arg1'] == 1
    assert captured_args['arg2'] == "two"
    assert captured_args['kwarg1'] == 3.0

def test_exception_wrapper_handler_with_var_kw():
    captured = {}
    def handler(e, arg, **kwargs):
        captured['arg'] = arg
        captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func(arg, kw1=None, kw2=None):
        raise Exception("error")
    func(42, kw1="value1", kw2="value2")
    assert captured['arg'] == 42
    assert captured['kwargs'] == {'kw1': 'value1', 'kw2': 'value2'}

def test_exception_wrapper_handler_missing_required_arg_raises():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_handler_with_default_matching_arg_raises():
    def handler(e, arg_with_default=5):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg_with_default):
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_handler_with_varargs_raises():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_generator():
    error_raised = False
    def handler(e):
        nonlocal error_raised
        error_raised = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert error_raised

def test_exception_wrapper_normal_return():
    @exception_wrapper()
    def normal_func():
        return 42
    result = normal_func()
    assert result == 42

def test_exception_wrapper_generator_normal():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_handler_receives_all_kwargs():
    captured_kwargs = {}
    def handler(e, **kwargs):
        captured_kwargs.update(kwargs)
    @exception_wrapper(handler)
    def func(a, b, c=3, d=4):
        raise Exception("error")
    func(1, b=2, d=5)
    assert captured_kwargs == {'a': 1, 'b': 2, 'c': 3, 'd': 5}

def test_exception_wrapper_handler_with_mixed_args():
    captured = {}
    def handler(e, a, b, c=30, **kwargs):
        captured['a'] = a
        captured['b'] = b
        captured['c'] = c
        captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func(a, b, c=3, d=4):
        raise Exception("error")
    func(10, b=20, d=40)
    assert captured['a'] == 10
    assert captured['b'] == 20
    assert captured['c'] == 3
    assert captured['kwargs'] == {'d': 40}

def test_exception_wrapper_wrapped_function():
    def inner():
        pass
    inner.__wrapped__ = lambda: None
    def handler(e):
        pass
    decorator = exception_wrapper(handler)
    wrapped = decorator(inner)
    assert hasattr(wrapped, '__wrapped__')


# LLM-generated content at query #24
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #25
#--------------------------

def test_register_ipython_excepthook_default():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    try:
        register_ipython_excepthook()
        assert sys.excepthook != original_hook
        assert callable(sys.excepthook)
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        assert sys.excepthook != original_hook
        assert callable(sys.excepthook)
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        assert sys.excepthook != original_hook
        assert callable(sys.excepthook)
    finally:
        sys.excepthook = original_hook


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def foo():
            pass
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #27
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    captured_exception = None
    captured_args = {}
    def custom_handler(e, arg1, arg2, extra=None):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "extra": extra}
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, extra=None):
        raise RuntimeError("custom error")
    func_with_args("value1", "value2", extra="extra_value")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["extra"] == "extra_value"

def test_exception_wrapper_handler_with_kwargs():
    captured_kwargs = {}
    def handler_with_kwargs(e, arg, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(arg, kwarg1=None, **extra_kwargs):
        raise Exception()
    func_with_kwargs("arg_value", kwarg1="kw1", extra1="ex1", extra2="ex2")
    assert captured_kwargs["kwarg1"] == "kw1"
    assert captured_kwargs["extra1"] == "ex1"
    assert captured_kwargs["extra2"] == "ex2"

def test_exception_wrapper_generator():
    error_raised = False
    def handler(e):
        nonlocal error_raised
        error_raised = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert error_raised

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_handler_missing_arg():
    def handler_missing(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler_missing)
        def func():
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches():
    def handler_default(e, arg_with_default=10):
        pass
    try:
        @exception_wrapper(handler_default)
        def func(arg_with_default):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_error():
    def handler_varargs(e, *args):
        pass
    try:
        @exception_wrapper(handler_varargs)
        def func():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args():
    def handler_no_args():
        pass
    try:
        @exception_wrapper(handler_no_args)
        def func():
            pass
    except ValueError as e:
        assert "must have a positional argument" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise ValueError("inner error")
    wrapped = exception_wrapper()(inner)
    wrapped()

def test_exception_wrapper_log_exception_called():
    @exception_wrapper()
    def func():
        raise ValueError("log test")
    func()


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    try:
        @decorator
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #29
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler_with_matching_args():
    handler_called = []
    def handler(e, arg1, arg2):
        handler_called.append((e, arg1, arg2))
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("custom error")
    func("a", "b")
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == "a"
    assert handler_called[0][2] == "b"

def test_exception_wrapper_custom_handler_with_default_args():
    handler_called = []
    def handler(e, arg1, arg2, extra=None):
        handler_called.append((e, arg1, arg2, extra))
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise TypeError("error with default")
    func(1, 2)
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], TypeError)
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2
    assert handler_called[0][3] is None

def test_exception_wrapper_custom_handler_with_kwargs():
    handler_called = []
    def handler(e, arg1, **kw):
        handler_called.append((e, arg1, kw))
    @exception_wrapper(handler)
    def func(arg1, arg2, **kwargs):
        raise ValueError("kwargs error")
    func("x", arg2="y", extra="z")
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == "x"
    assert handler_called[0][2] == {"arg2": "y", "kwargs": {"extra": "z"}}

def test_exception_wrapper_generator_function():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
    gen = gen_func()
    result = list(gen)
    assert result == [1]

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func():
        return "success"
    result = normal_func()
    assert result == "success"

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    gen = gen_func()
    result = list(gen)
    assert result == [0, 1, 2]

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_arg_not_in_wrapped():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches_wrapped():
    try:
        def handler(e, arg1, arg2=10):
            pass
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
        func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function_with_defaults():
    handler_called = []
    def handler(e, arg1):
        handler_called.append((e, arg1))
    @exception_wrapper(handler)
    def func(arg1, arg2="default"):
        raise KeyError("default arg error")
    func("value")
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], KeyError)
    assert handler_called[0][1] == "value"

def test_exception_wrapper_wrapped_function_with_args_kwargs():
    handler_called = []
    def handler(e, arg1, **kw):
        handler_called.append((e, arg1, kw))
    @exception_wrapper(handler)
    def func(arg1, *args, **kwargs):
        raise IndexError("args kwargs error")
    func("first", "extra", extra_kw="val")
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], IndexError)
    assert handler_called[0][1] == "first"
    assert handler_called[0][2] == {"args": ("extra",), "kwargs": {"extra_kw": "val"}}

def test_exception_wrapper_already_wrapped_function():
    def inner_handler(e, arg):
        pass
    @exception_wrapper(inner_handler)
    def inner(arg):
        raise Exception("inner")
    outer_handler_called = []
    def outer_handler(e, arg):
        outer_handler_called.append((e, arg))
    wrapped = exception_wrapper(outer_handler)(inner)
    wrapped("test")
    assert len(outer_handler_called) == 1
    assert isinstance(outer_handler_called[0][0], Exception)
    assert outer_handler_called[0][1] == "test"


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #31
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def failing_function():
        raise ValueError("test error")
    failing_function()

def test_exception_wrapper_passes_through_return_value():
    @exception_wrapper()
    def successful_function():
        return 42
    result = successful_function()
    assert result == 42

def test_exception_wrapper_wraps_generator():
    @exception_wrapper()
    def generator_function():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = generator_function()
    assert list(gen) == [1]

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def failing_function():
        raise RuntimeError("custom error")
    failing_function()
    assert isinstance(caught_exception, RuntimeError)
    assert str(caught_exception) == "custom error"

def test_exception_wrapper_handler_with_matching_args():
    captured_args = {}
    def custom_handler(e, arg1, arg2):
        nonlocal captured_args
        captured_args = {"e": e, "arg1": arg1, "arg2": arg2}
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("error with args")
    failing_function("value1", arg2="value2")
    assert isinstance(captured_args["e"], ValueError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"

def test_exception_wrapper_handler_with_default_args():
    captured_args = {}
    def custom_handler(e, arg1, default_arg="default"):
        nonlocal captured_args
        captured_args = {"e": e, "arg1": arg1, "default_arg": default_arg}
    @exception_wrapper(custom_handler)
    def failing_function(arg1):
        raise ValueError("error with default")
    failing_function("value1")
    assert isinstance(captured_args["e"], ValueError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["default_arg"] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured_args = {}
    def custom_handler(e, arg1, **kwargs):
        nonlocal captured_args
        captured_args = {"e": e, "arg1": arg1, "kwargs": kwargs}
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2, arg3=None):
        raise ValueError("error with kwargs")
    failing_function("value1", "value2", arg3="value3")
    assert isinstance(captured_args["e"], ValueError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["kwargs"] == {"arg2": "value2", "arg3": "value3"}

def test_exception_wrapper_handler_missing_arg_raises():
    def custom_handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_matched_arg_with_default_raises():
    def custom_handler(e, arg1, default_arg="default"):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(arg1, default_arg):
            pass
        assert False
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_raises():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args_raises():
    def custom_handler():
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "must have a positional argument" in str(e)

def test_exception_wrapper_nested_wrapping():
    call_count = 0
    def custom_handler(e):
        nonlocal call_count
        call_count += 1
    @exception_wrapper(custom_handler)
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("nested error")
    failing_function()
    assert call_count == 1


# LLM-generated content at query #32
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def failing_function():
        raise ValueError("test error")
    failing_function()

def test_exception_wrapper_custom_handler():
    captured_exception = None
    def custom_handler(e):
        nonlocal captured_exception
        captured_exception = e
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("custom error")
    failing_function()
    assert captured_exception is not None
    assert str(captured_exception) == "custom error"

def test_exception_wrapper_passes_arguments_to_handler():
    captured_args = {}
    def custom_handler(e, arg1, arg2, kwarg1=None):
        nonlocal captured_args
        captured_args = {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1}
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2, kwarg1=None):
        raise RuntimeError("args test")
    failing_function("value1", "value2", kwarg1="value3")
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["kwarg1"] == "value3"

def test_exception_wrapper_handler_with_kwargs():
    captured_kwargs = {}
    def custom_handler(e, arg, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = {"arg": arg, "kwargs": kwargs}
    @exception_wrapper(custom_handler)
    def failing_function(arg, extra=5):
        raise Exception("kwargs test")
    failing_function("test_arg", extra=10)
    assert captured_kwargs["arg"] == "test_arg"
    assert captured_kwargs["kwargs"] == {"extra": 10}

def test_exception_wrapper_returns_value():
    @exception_wrapper()
    def successful_function():
        return 42
    result = successful_function()
    assert result == 42

def test_exception_wrapper_generator_exception():
    exception_logged = False
    def custom_handler(e):
        nonlocal exception_logged
        exception_logged = True
    @exception_wrapper(custom_handler)
    def failing_generator():
        yield 1
        raise ValueError("generator error")
    gen = failing_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert exception_logged

def test_exception_wrapper_generator_yields_values():
    @exception_wrapper()
    def successful_generator():
        yield 1
        yield 2
    gen = successful_generator()
    assert list(gen) == [1, 2]

def test_exception_wrapper_handler_missing_arg_raises_error():
    def custom_handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_default_arg_matches_raises_error():
    def custom_handler(e, arg_with_default=5):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(arg_with_default):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_raises_error():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_no_handler_arg_raises_error():
    def custom_handler():
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
    except ValueError as e:
        assert "must have a positional argument" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        pass
    inner.__wrapped__ = lambda: None
    def custom_handler(e):
        pass
    decorator = exception_wrapper(custom_handler)
    wrapped = decorator(inner)

def test_exception_wrapper_log_exception_integration():
    @exception_wrapper()
    def function_raising_called_process_error():
        error = subprocess.CalledProcessError(1, "cmd")
        error.output = b"output"
        raise error
    function_raising_called_process_error()

def test_exception_wrapper_log_exception_user_msg():
    @exception_wrapper()
    def function_with_log_exception():
        try:
            raise ValueError("inner")
        except ValueError as e:
            log_exception(e, user_msg="User message")
    function_with_log_exception()


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_handler_arg_without_default_matches_wrapped():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    func(1, 2)


# LLM-generated content at query #34
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_false():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    excepthook = sys.excepthook
    mock_type = KeyboardInterrupt
    mock_value = KeyboardInterrupt()
    mock_traceback = Mock(spec=TracebackType)
    with patch('sys.__excepthook__') as mock_sys_excepthook:
        excepthook(mock_type, mock_value, mock_traceback)
        mock_sys_excepthook.assert_called_once_with(mock_type, mock_value, mock_traceback)

def test_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    from IPython.core.ultratb import FormattedTB
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    excepthook = sys.excepthook
    mock_type = KeyboardInterrupt
    mock_value = KeyboardInterrupt()
    mock_traceback = Mock(spec=TracebackType)
    with patch.object(FormattedTB, '__call__') as mock_ipython_hook:
        excepthook(mock_type, mock_value, mock_traceback)
        mock_ipython_hook.assert_called_once_with(mock_type, mock_value, mock_traceback)


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #36
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    captured_exception = None
    captured_args = {}
    def handler(e, arg1, arg2, extra=None):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "extra": extra}
    @exception_wrapper(handler)
    def func(arg1, arg2, extra=None):
        raise RuntimeError("custom error")
    func("value1", "value2", extra="extra_value")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["extra"] == "extra_value"

def test_exception_wrapper_with_generator():
    log_messages = []
    original_log = log
    def mock_log(msg, level="info", **kwargs):
        log_messages.append((msg, level))
    import flutes.log
    flutes.log.log = mock_log
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    result = list(gen_func())
    flutes.log.log = original_log
    assert result == [1]
    assert any("generator error" in msg for msg, level in log_messages if level == "error")

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg, **kwargs):
        captured["e"] = e
        captured["arg"] = arg
        captured["kwargs"] = kwargs
    @exception_wrapper(handler)
    def func(arg, extra=5):
        raise Exception("test")
    func("test_arg", extra=10)
    assert isinstance(captured["e"], Exception)
    assert captured["arg"] == "test_arg"
    assert captured["kwargs"] == {"extra": 10}

def test_exception_wrapper_handler_missing_arg():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
        func()
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches():
    def handler(e, arg, extra=None):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg, extra):
            pass
        func("test", extra=5)
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        exception_wrapper(handler)
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args():
    def handler():
        pass
    try:
        exception_wrapper(handler)
    except ValueError as e:
        assert "must have a positional argument" in str(e)

def test_exception_wrapper_normal_return():
    @exception_wrapper()
    def func():
        return 42
    assert func() == 42

def test_exception_wrapper_generator_normal():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    assert list(gen_func()) == [0, 1, 2]

def test_exception_wrapper_wrapped_function():
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper()
    @decorator
    def func():
        raise ValueError("wrapped error")
    func()


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def foo():
            pass
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #38
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_handler_with_matching_args():
    handler_args = {}
    def custom_handler(e, arg1, arg2):
        nonlocal handler_args
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2}
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, arg2="hello")
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 10
    assert handler_args["arg2"] == "hello"

def test_exception_wrapper_handler_with_default_args():
    handler_args = {}
    def custom_handler(e, arg1, my_default=5):
        nonlocal handler_args
        handler_args = {"e": e, "arg1": arg1, "my_default": my_default}
    @exception_wrapper(custom_handler)
    def func(arg1):
        raise TypeError("type error")
    func(42)
    assert isinstance(handler_args["e"], TypeError)
    assert handler_args["arg1"] == 42
    assert handler_args["my_default"] == 5

def test_exception_wrapper_handler_with_kwargs():
    handler_args = {}
    def custom_handler(e, arg1, **kw):
        nonlocal handler_args
        handler_args = {"e": e, "arg1": arg1, "kw": kw}
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, extra=3):
        raise ValueError("val error")
    func(1, arg2=2, extra=4, additional=5)
    assert isinstance(handler_args["e"], ValueError)
    assert handler_args["arg1"] == 1
    assert handler_args["kw"] == {"arg2": 2, "extra": 4, "additional": 5}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_raise(x):
        return x * 2
    result = func_no_raise(21)
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func(items):
        for item in items:
            yield item * 2
    gen = gen_func([1, 2, 3])
    results = list(gen)
    assert results == [2, 4, 6]

def test_exception_wrapper_generator_with_exception():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def gen_func(items):
        for item in items:
            if item == 2:
                raise ValueError("bad item")
            yield item * 2
    gen = gen_func([1, 2, 3])
    results = list(gen)
    assert results == [2]
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "bad item"

def test_exception_wrapper_handler_missing_arg_raises():
    def custom_handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def func():
            pass
        assert False
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_matching_arg_with_default_raises():
    def custom_handler(e, arg1, default_arg=10):
        pass
    try:
        @exception_wrapper(custom_handler)
        def func(arg1):
            pass
        assert False
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_raises():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def func():
            pass
        assert False
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args_raises():
    def custom_handler():
        pass
    try:
        @exception_wrapper(custom_handler)
        def func():
            pass
        assert False
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #39
#--------------------------

def test_register_ipython_excepthook_default():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    assert callable(sys.excepthook)

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    assert callable(sys.excepthook)

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    from unittest.mock import Mock, patch
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    assert callable(sys.excepthook)

def test_excepthook_skips_bdbquit():
    import sys
    from unittest.mock import Mock, patch
    register_ipython_excepthook()
    mock_original_hook = Mock()
    with patch('sys.__excepthook__', mock_original_hook):
        sys.excepthook(BdbQuit, BdbQuit(), None)
    mock_original_hook.assert_called_once()

def test_excepthook_skips_keyboard_interrupt_by_default():
    import sys
    from unittest.mock import Mock, patch
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    mock_original_hook = Mock()
    with patch('sys.__excepthook__', mock_original_hook):
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    mock_original_hook.assert_called_once()

def test_excepthook_captures_keyboard_interrupt_when_enabled():
    import sys
    from unittest.mock import Mock, patch
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    mock_ipython_hook = Mock()
    with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    mock_ipython_hook.assert_called_once()

def test_excepthook_calls_ipython_hook_for_other_exceptions():
    import sys
    from unittest.mock import Mock, patch
    mock_ipython_hook = Mock()
    with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
        register_ipython_excepthook()
        sys.excepthook(ValueError, ValueError("test"), None)
    mock_ipython_hook.assert_called_once()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_log_exception_with_user_message():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")
    logged_msg = None
    for handler in log.handlers:
        if hasattr(handler, 'messages'):
            logged_msg = handler.messages[-1]
            break
    assert logged_msg is not None
    assert "Custom message" in logged_msg
    assert "ValueError" in logged_msg
    assert "test error" in logged_msg

def test_log_exception_without_user_message():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise RuntimeError("runtime test")
    except RuntimeError as e:
        log_exception(e)
    logged_msg = None
    for handler in log.handlers:
        if hasattr(handler, 'messages'):
            logged_msg = handler.messages[-1]
            break
    assert logged_msg is not None
    assert "RuntimeError" in logged_msg
    assert "runtime test" in logged_msg

def test_log_exception_with_called_process_error():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    logged_msg = None
    for handler in log.handlers:
        if hasattr(handler, 'messages'):
            logged_msg = handler.messages[-1]
            break
    assert logged_msg is not None
    assert "CalledProcessError" in logged_msg
    assert "Command '['ls']' returned non-zero exit status 1." in logged_msg

def test_log_exception_logging_failure():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    original_log = log
    def failing_log(*args, **kwargs):
        raise IOError("log failed")
    log = failing_log
    try:
        raise KeyError("key missing")
    except KeyError as e:
        try:
            log_exception(e)
        except IOError as log_e:
            assert str(log_e) == "log failed"
    log = original_log


# LLM-generated content at query #2
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #3
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_raises():
        raise ValueError("test error")
    func_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured_args = {}
    def handler(e, one, two):
        nonlocal captured_args
        captured_args = {"one": one, "two": two}
    @exception_wrapper(handler)
    def func(one, two):
        raise RuntimeError("error")
    func(1, 2)
    assert captured_args == {"one": 1, "two": 2}

def test_exception_wrapper_custom_handler_with_default_args():
    captured = {}
    def handler(e, one, my_arg="default"):
        nonlocal captured
        captured = {"one": one, "my_arg": my_arg}
    @exception_wrapper(handler)
    def func(one):
        raise Exception()
    func(1)
    assert captured == {"one": 1, "my_arg": "default"}

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def handler(e, one, **kw):
        nonlocal captured
        captured = {"one": one, "kw": kw}
    @exception_wrapper(handler)
    def func(one, two, three=3):
        raise Exception()
    func(1, two=2)
    assert captured["one"] == 1
    assert captured["kw"] == {"two": 2, "three": 3}

def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    g = gen_func()
    assert next(g) == 1
    try:
        next(g)
    except StopIteration:
        pass

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_nested_wrapping():
    @exception_wrapper()
    @exception_wrapper()
    def func():
        raise ValueError("nested")
    func()

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_handler_missing_arg_error():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "missing_arg" in str(e)

def test_exception_wrapper_handler_default_arg_matches_error():
    def handler(e, arg, default_arg=5):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg, default_arg):
            pass
    except ValueError as e:
        assert "default_arg" in str(e)


# LLM-generated content at query #4
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_raises():
        raise ValueError("test error")
    func_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_handler_with_matching_args():
    args_captured = []
    def handler(e, arg1, arg2):
        args_captured.extend([arg1, arg2])
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise RuntimeError("error")
    func_raises(10, "hello")
    assert args_captured == [10, "hello"]

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def handler(e, arg1, my_arg="default"):
        captured["arg1"] = arg1
        captured["my_arg"] = my_arg
    @exception_wrapper(handler)
    def func_raises(arg1):
        raise Exception("error")
    func_raises(42)
    assert captured["arg1"] == 42
    assert captured["my_arg"] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg1, **kw):
        captured["arg1"] = arg1
        captured["kw"] = kw
    @exception_wrapper(handler)
    def func_raises(arg1, arg2=None, **kwargs):
        raise Exception("error")
    func_raises(1, arg2=2, extra=3)
    assert captured["arg1"] == 1
    assert captured["kw"] == {"arg2": 2, "kwargs": {"extra": 3}}

def test_exception_wrapper_generator():
    error_raised = False
    def handler(e):
        nonlocal error_raised
        error_raised = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert error_raised

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_missing_handler_arg():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def dummy(arg1):
            pass
    except ValueError as e:
        assert "Argument 'missing_arg' in exception handler does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches():
    try:
        def handler(e, arg1, matched_arg="default"):
            pass
        @exception_wrapper(handler)
        def dummy(matched_arg):
            pass
    except ValueError as e:
        assert "Argument 'matched_arg' matches wrapped method argument" in str(e)


# LLM-generated content at query #5
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    original_excepthook = sys.__excepthook__
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    excepthook = sys.excepthook
    mock_ipython_hook = Mock()
    with patch('sys.__excepthook__', original_excepthook), patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
        excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        mock_ipython_hook.assert_not_called()


# LLM-generated content at query #6
#--------------------------

def test_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_is_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    skip_exceptions = [BdbQuit, KeyboardInterrupt]
    assert KeyboardInterrupt in skip_exceptions

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    skip_exceptions = [BdbQuit]
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)(lambda: None)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #8
#--------------------------

def test_log_exception_does_not_log_traceback_for_called_process_error_with_output():
    import subprocess
    import traceback
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd=["test"], output=b"output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with(f"<CalledProcessError> Command '['test']' returned non-zero exit status 1.", "error")
        mock_log.assert_any_call(traceback.format_exc(), "error")
        assert mock_log.call_count == 1


# LLM-generated content at query #9
#--------------------------

def test_log_exception_with_user_msg():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="User message")
    assert True

def test_log_exception_without_user_msg():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e)
    assert True

def test_log_exception_with_called_process_error():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    assert True

def test_log_exception_logging_failure():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    original_log = log
    def failing_log(*args, **kwargs):
        raise RuntimeError("log failed")
    log = failing_log
    try:
        raise ValueError("test error")
    except ValueError as e:
        try:
            log_exception(e)
        except RuntimeError as log_e:
            assert str(log_e) == "log failed"
    log = original_log


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_handler_without_positional_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def func():
        pass
    try:
        decorator(func)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #11
#--------------------------

def test_log_exception_with_user_message():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")

def test_log_exception_without_user_message():
    try:
        raise RuntimeError("Runtime issue")
    except RuntimeError as e:
        log_exception(e)

def test_log_exception_with_additional_kwargs():
    try:
        raise KeyError("Missing key")
    except KeyError as e:
        log_exception(e, force_console=True, timestamp=False)

def test_log_exception_called_process_error_with_output():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

def test_log_exception_called_process_error_without_output():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=None)
    except subprocess.CalledProcessError as e:
        log_exception(e)

def test_log_exception_logging_failure():
    original_log = log
    log = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Log failure"))
    try:
        try:
            raise TypeError("Type mismatch")
        except TypeError as e:
            log_exception(e)
    except Exception as log_e:
        assert str(log_e) == "Log failure"
    finally:
        log = original_log


# LLM-generated content at query #12
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #13
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")

    func_raises()

def test_exception_wrapper_with_custom_handler():
    captured_exception = None
    captured_args = {}

    def custom_handler(e, arg1, arg2, kwarg1=None):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, kwarg1="default"):
        raise RuntimeError("custom error")

    func_with_args("value1", "value2", kwarg1="custom")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["kwarg1"] == "custom"

def test_exception_wrapper_with_matching_args():
    handler_called = False

    def custom_handler(e, param):
        nonlocal handler_called
        handler_called = True
        assert param == "test_param"

    @exception_wrapper(custom_handler)
    def func_single_param(param):
        raise Exception()

    func_single_param("test_param")
    assert handler_called

def test_exception_wrapper_with_kwargs():
    handler_kwargs = {}

    def custom_handler(e, **kwargs):
        nonlocal handler_kwargs
        handler_kwargs = kwargs

    @exception_wrapper(custom_handler)
    def func_with_kwargs(a, b=2):
        raise Exception()

    func_with_kwargs(1, b=3)
    assert handler_kwargs["a"] == 1
    assert handler_kwargs["b"] == 3

def test_exception_wrapper_with_args_and_kwargs():
    handler_args = {}

    def custom_handler(e, first, second, extra=None, **kwargs):
        nonlocal handler_args
        handler_args = {"first": first, "second": second, "extra": extra, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_mixed(first, *args, second=20, **kwargs):
        raise Exception()

    func_mixed(10, "arg1", second=30, extra_kw="value")
    assert handler_args["first"] == 10
    assert handler_args["second"] == 30
    assert handler_args["extra"] is None
    assert handler_args["kwargs"]["args"] == ("arg1",)
    assert handler_args["kwargs"]["extra_kw"] == "value"

def test_exception_wrapper_with_generator():
    error_raised = False

    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        error_raised = True
    assert error_raised

def test_exception_wrapper_with_generator_and_handler():
    handler_called = False

    def custom_handler(e):
        nonlocal handler_called
        handler_called = True

    @exception_wrapper(custom_handler)
    def gen_func_with_args(x):
        yield x
        raise RuntimeError()

    gen = gen_func_with_args(5)
    assert next(gen) == 5
    try:
        next(gen)
    except StopIteration:
        pass
    assert handler_called

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func():
        return 42

    result = normal_func()
    assert result == 42

def test_exception_wrapper_with_nested_decorator():
    call_count = 0

    def custom_handler(e):
        nonlocal call_count
        call_count += 1

    @exception_wrapper(custom_handler)
    @exception_wrapper(custom_handler)
    def double_wrapped():
        raise IndexError()

    double_wrapped()
    assert call_count == 1

def test_exception_wrapper_invalid_handler_no_args():
    try:
        @exception_wrapper(lambda: None)
        def dummy():
            pass
        dummy()
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_varargs():
    try:
        @exception_wrapper(lambda e, *args: None)
        def dummy():
            pass
        dummy()
    except ValueError:
        pass

def test_exception_wrapper_missing_handler_arg():
    try:
        @exception_wrapper(lambda e, missing_arg: None)
        def dummy():
            pass
        dummy()
    except ValueError:
        pass

def test_exception_wrapper_handler_arg_with_default_matches():
    try:
        @exception_wrapper(lambda e, arg="default": None)
        def dummy(arg):
            pass
        dummy()
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")


# LLM-generated content at query #15
#--------------------------

def test_log_exception_with_non_called_process_error():
    e = ValueError("test error")
    log_exception(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #17
#--------------------------

def test_register_ipython_excepthook_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_is_false():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    class BdbQuit(BaseException):
        pass
    class KeyboardInterrupt(BaseException):
        pass
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    type_mock = Mock(spec=Type[BaseException])
    value_mock = Mock()
    traceback_mock = Mock(spec=TracebackType)
    any_result = any(type_mock is exc_type for exc_type in skip_exceptions)
    assert any_result == False


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_fn():
        pass
    try:
        exception_wrapper(handler_fn)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #19
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped_method_arg():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    func(1, 2)


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def foo():
            pass
        foo()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_handler_arg_names_without_defaults():
    def handler_fn(e, arg1, arg2, kwarg1=None, kwarg2=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None, kwarg2=None):
        pass

    func(1, 2, kwarg1="a", kwarg2="b")
    assert True


# LLM-generated content at query #23
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()

def test_exception_wrapper_custom_handler():
    captured_exception = None
    def handler(e):
        nonlocal captured_exception
        captured_exception = e
    @exception_wrapper(handler)
    def func():
        raise ValueError("test error")
    func()
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured_args = {}
    def handler(e, arg1, arg2):
        nonlocal captured_args
        captured_args = {"arg1": arg1, "arg2": arg2}
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, "hello")
    assert captured_args["arg1"] == 10
    assert captured_args["arg2"] == "hello"

def test_exception_wrapper_custom_handler_with_default_args():
    captured_args = {}
    def handler(e, arg1, arg2, optional="default"):
        nonlocal captured_args
        captured_args = {"arg1": arg1, "arg2": arg2, "optional": optional}
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, "world")
    assert captured_args["arg1"] == 5
    assert captured_args["arg2"] == "world"
    assert captured_args["optional"] == "default"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_kwargs = {}
    def handler(e, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(handler)
    def func(a, b, c=3):
        raise ValueError("error")
    func(1, b=2)
    assert captured_kwargs["a"] == 1
    assert captured_kwargs["b"] == 2
    assert captured_kwargs["c"] == 3

def test_exception_wrapper_custom_handler_with_mixed_args():
    captured = {}
    def handler(e, x, y, z=30, **extra):
        nonlocal captured
        captured = {"x": x, "y": y, "z": z, "extra": extra}
    @exception_wrapper(handler)
    def func(x, y, z=10):
        raise RuntimeError("error")
    func(100, y=200)
    assert captured["x"] == 100
    assert captured["y"] == 200
    assert captured["z"] == 10
    assert captured["extra"] == {}

def test_exception_wrapper_generator_function():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_generator_function_custom_handler():
    caught = False
    def handler(e):
        nonlocal caught
        caught = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert caught

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func():
        return 42
    result = func()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_handler_with_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_handler_no_exception_arg_error():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e)

def test_exception_wrapper_handler_missing_arg_error():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_default_arg_matches_error():
    def handler(e, arg1, arg2="default"):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_handler_arg_matches_wrapped_method_arg_with_default_value():
    def handler_fn(e, arg_with_default):
        pass

    @exception_wrapper(handler_fn)
    def wrapped_func(arg_with_default=42):
        pass

    try:
        wrapped_func()
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #25
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")


# LLM-generated content at query #26
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()

def test_exception_wrapper_custom_handler():
    captured = []
    def handler(e, x):
        captured.append((e, x))
    @exception_wrapper(handler)
    def func(x):
        raise RuntimeError("error")
    func(5)
    assert len(captured) == 1
    assert isinstance(captured[0][0], RuntimeError)
    assert captured[0][1] == 5

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = []
    def handler(e, a, b, extra=None):
        captured.append((e, a, b, extra))
    @exception_wrapper(handler)
    def func(a, b, c=10):
        raise KeyError("key")
    func(1, 2)
    assert len(captured) == 1
    assert isinstance(captured[0][0], KeyError)
    assert captured[0][1] == 1
    assert captured[0][2] == 2
    assert captured[0][3] is None

def test_exception_wrapper_custom_handler_with_varkw():
    captured = []
    def handler(e, x, **kw):
        captured.append((e, x, kw))
    @exception_wrapper(handler)
    def func(x, y, z=3):
        raise TypeError("type")
    func(7, y=8)
    assert len(captured) == 1
    assert isinstance(captured[0][0], TypeError)
    assert captured[0][1] == 7
    assert captured[0][2] == {'y': 8, 'z': 3}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func():
        return 42
    result = func()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def func():
        yield from range(3)
    gen = func()
    assert list(gen) == [0, 1, 2]

def test_exception_wrapper_generator_with_exception():
    captured = []
    def handler(e):
        captured.append(e)
    @exception_wrapper(handler)
    def func():
        yield 1
        raise ValueError("gen error")
        yield 2
    gen = func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert len(captured) == 1
    assert isinstance(captured[0], ValueError)

def test_exception_wrapper_handler_missing_arg():
    def handler(e, missing):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches():
    def handler(e, x, y=10):
        pass
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "must have a positional argument" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise IndexError("inner")
    @exception_wrapper()
    @functools.wraps(inner)
    def func():
        inner()
    func()

def test_exception_wrapper_log_exception_called():
    @exception_wrapper()
    def func():
        raise Exception("log me")
    func()


# LLM-generated content at query #27
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd='test', output=b'output')
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with('<CalledProcessError> Command \'test\' returned non-zero exit status 1.', 'error')
    e = subprocess.CalledProcessError(returncode=1, cmd='test', output=None)
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), 'error')
        mock_log.assert_any_call('<CalledProcessError> Command \'test\' returned non-zero exit status 1.', 'error')
    e = ValueError('test')
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), 'error')
        mock_log.assert_any_call('<ValueError> test', 'error')


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #29
#--------------------------

def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_handler():
    caught_exception = None
    handler_args = {}

    def handler(e, arg1, arg2, optional_arg="default"):
        nonlocal caught_exception, handler_args
        caught_exception = e
        handler_args = {"arg1": arg1, "arg2": arg2, "optional_arg": optional_arg}

    @exception_wrapper(handler)
    def func(arg1, arg2, optional_arg="default"):
        raise RuntimeError("error inside")

    func("value1", "value2", optional_arg="custom")
    assert isinstance(caught_exception, RuntimeError)
    assert str(caught_exception) == "error inside"
    assert handler_args["arg1"] == "value1"
    assert handler_args["arg2"] == "value2"
    assert handler_args["optional_arg"] == "custom"


def test_exception_wrapper_with_handler_and_kwargs():
    caught_exception = None
    handler_kwargs = {}

    def handler(e, arg, **kwargs):
        nonlocal caught_exception, handler_kwargs
        caught_exception = e
        handler_kwargs = kwargs

    @exception_wrapper(handler)
    def func(arg, extra="extra_default", **kwargs):
        raise TypeError("type error")

    func("arg_value", extra="extra_value", additional="additional_value")
    assert isinstance(caught_exception, TypeError)
    assert str(caught_exception) == "type error"
    assert handler_kwargs["arg"] == "arg_value"
    assert handler_kwargs["extra"] == "extra_value"
    assert handler_kwargs["additional"] == "additional_value"


def test_exception_wrapper_with_generator():
    log_messages = []

    def mock_log(msg, level, **kwargs):
        log_messages.append((msg, level))

    global log
    original_log = log
    log = mock_log

    @exception_wrapper()
    def generator_func():
        yield 1
        raise ValueError("generator error")
        yield 2

    gen = generator_func()
    result = list(gen)
    log = original_log
    assert result == [1]
    assert len(log_messages) == 2
    assert log_messages[0][1] == "error"
    assert "Traceback" in log_messages[0][0]
    assert log_messages[1][1] == "error"
    assert "<ValueError> generator error" in log_messages[1][0]


def test_exception_wrapper_with_nested_decorator():
    call_count = 0

    def handler(e):
        nonlocal call_count
        call_count += 1

    @exception_wrapper(handler)
    @exception_wrapper(handler)
    def func():
        raise Exception("nested")

    func()
    assert call_count == 1


def test_exception_wrapper_handler_without_exception():
    @exception_wrapper()
    def normal_func():
        return "success"

    result = normal_func()
    assert result == "success"


def test_exception_wrapper_handler_with_generator_no_exception():
    @exception_wrapper()
    def generator_func():
        yield from range(3)

    result = list(generator_func())
    assert result == [0, 1, 2]


def test_exception_wrapper_invalid_handler_varargs():
    def invalid_handler(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)


def test_exception_wrapper_invalid_handler_no_args():
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)


def test_exception_wrapper_handler_arg_mismatch():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)


def test_exception_wrapper_handler_default_arg_matches():
    def handler(e, existing_arg="default"):
        pass

    try:
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #30
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped_arg():
    from flutes.exception import exception_wrapper
    import inspect

    def handler_fn(e, arg1, arg2=5):
        pass

    @exception_wrapper(handler_fn)
    def wrapped_func(arg1, arg2):
        pass

    try:
        inspect.signature(wrapped_func)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_handler_arg_without_matching_name_must_have_default():
    def handler_fn(e, my_arg=None):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two):
        pass

    foo(1, 2)


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception():
        pass

    @exception_wrapper(handler_without_exception)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_value_error():
    def handler_with_varargs(e, *args):
        pass
    decorator = exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #36
#--------------------------

def test_exception_wrapper_logs_exception_with_default_handler():
    @exception_wrapper()
    def failing_function():
        raise ValueError("test error")
    failing_function()

def test_exception_wrapper_passes_exception_to_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("test error")
    failing_function()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_custom_handler_receives_matching_args():
    received_args = {}
    def custom_handler(e, arg1, arg2):
        received_args['arg1'] = arg1
        received_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("test error")
    failing_function(10, arg2=20)
    assert received_args['arg1'] == 10
    assert received_args['arg2'] == 20

def test_exception_wrapper_custom_handler_receives_kwargs():
    received_kwargs = {}
    def custom_handler(e, **kwargs):
        received_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("test error")
    failing_function(10, arg2=20)
    assert received_kwargs['arg1'] == 10
    assert received_kwargs['arg2'] == 20

def test_exception_wrapper_custom_handler_with_default_args():
    received_args = {}
    def custom_handler(e, arg1, arg2, default_arg="default"):
        received_args['arg1'] = arg1
        received_args['arg2'] = arg2
        received_args['default_arg'] = default_arg
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("test error")
    failing_function(10, arg2=20)
    assert received_args['arg1'] == 10
    assert received_args['arg2'] == 20
    assert received_args['default_arg'] == "default"

def test_exception_wrapper_raises_error_on_varargs_in_handler():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
    except ValueError as e:
        assert "varargs" in str(e).lower() or "*args" in str(e)

def test_exception_wrapper_raises_error_on_missing_handler_arg():
    def custom_handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(existing_arg):
            pass
    except ValueError as e:
        assert "missing_arg" in str(e) or "does not match" in str(e)

def test_exception_wrapper_raises_error_on_default_arg_matching_wrapped():
    def custom_handler(e, arg1, default_arg="default"):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(arg1, default_arg):
            pass
    except ValueError as e:
        assert "default_arg" in str(e) and "cannot have default values" in str(e)

def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def successful_function():
        return 42
    result = successful_function()
    assert result == 42

def test_exception_wrapper_preserves_generator():
    @exception_wrapper()
    def generator_function():
        yield 1
        yield 2
    gen = generator_function()
    assert list(gen) == [1, 2]

def test_exception_wrapper_catches_exception_in_generator():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def failing_generator():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = failing_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_works_with_wrapped_functions():
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @decorator
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("test error")
    failing_function()
    assert isinstance(caught_exception, ValueError)

def test_exception_wrapper_handler_receives_bound_args_with_defaults():
    received_args = {}
    def custom_handler(e, arg1, arg2):
        received_args['arg1'] = arg1
        received_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2=100):
        raise ValueError("test error")
    failing_function(10)
    assert received_args['arg1'] == 10
    assert received_args['arg2'] == 100


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_arg():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def dummy():
        pass

    try:
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)(lambda: None)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_error():
    def handler_with_varargs(e, *args):
        pass
    decorator = exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #41
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    result = register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #42
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from bdb import BdbQuit
    from typing import Type, List
    from IPython.core.ultratb import FormattedTB

    def register_ipython_excepthook(capture_keyboard_interrupt: bool = False) -> None:
        skip_exceptions: List[Type[BaseException]] = [BdbQuit]
        if not capture_keyboard_interrupt:
            skip_exceptions.append(KeyboardInterrupt)

        def excepthook(type, value, traceback):
            if any(type is exc_type for exc_type in skip_exceptions):
                sys.__excepthook__(type, value, traceback)
            else:
                ipython_hook(type, value, traceback)

        from IPython.core import ultratb
        ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
        sys.excepthook = excepthook

    register_ipython_excepthook(capture_keyboard_interrupt=True)
    excepthook_func = sys.excepthook
    mock_ipython_hook = Mock()
    with patch('sys.__excepthook__', Mock()) as mock_sys_excepthook:
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
            excepthook_func(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_sys_excepthook.assert_not_called()
            mock_ipython_hook.assert_called_once_with(KeyboardInterrupt, KeyboardInterrupt(), None)


# LLM-generated content at query #43
#--------------------------

def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_handler():
    captured_exception = None
    captured_args = {}

    def handler(e, arg1, arg2, extra="default"):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "extra": extra}

    @exception_wrapper(handler)
    def func_with_args(arg1, arg2, extra="default"):
        raise RuntimeError("error in func")

    func_with_args("value1", "value2")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["extra"] == "default"


def test_exception_wrapper_with_kwargs():
    captured_kwargs = {}

    def handler(e, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs

    @exception_wrapper(handler)
    def func_with_kwargs(a, b=2, **kwargs):
        raise Exception()

    func_with_kwargs(1, c=3)
    assert captured_kwargs["a"] == 1
    assert captured_kwargs["b"] == 2
    assert captured_kwargs["c"] == 3


def test_exception_wrapper_with_generator():
    error_occurred = False

    def handler(e):
        nonlocal error_occurred
        error_occurred = True

    @exception_wrapper(handler)
    def generator_func():
        yield 1
        raise ValueError("generator error")
        yield 2

    gen = generator_func()
    assert list(gen) == [1]
    assert error_occurred


def test_exception_wrapper_handler_missing_arg():
    def handler(e, missing_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass


def test_exception_wrapper_handler_arg_with_default_matches():
    def handler(e, arg_with_default="default"):
        pass

    try:
        @exception_wrapper(handler)
        def func(arg_with_default="default"):
            pass
        assert False
    except ValueError:
        pass


def test_exception_wrapper_handler_varargs():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass


def test_exception_wrapper_nested_wrapped():
    def handler(e, arg):
        pass

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    @exception_wrapper(handler)
    @decorator
    def func(arg):
        raise Exception()

    func("test")


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_arg():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_argument():
    try:
        @exception_wrapper(lambda: None)
        def foo():
            pass
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_handler_arg_matches_wrapped_method_arg_with_default():
    def handler_fn(e, arg_with_default):
        pass

    @exception_wrapper(handler_fn)
    def func(arg_with_default=42):
        pass

    try:
        func()
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #49
#--------------------------

def test_register_ipython_excepthook_default():
    import sys
    from unittest.mock import patch, MagicMock
    original_hook = sys.excepthook
    try:
        register_ipython_excepthook()
        assert sys.excepthook != original_hook
        mock_ultratb = MagicMock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb):
            register_ipython_excepthook()
            sys.excepthook(ValueError, ValueError("test"), None)
            mock_ultratb.assert_not_called()
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    from unittest.mock import patch, MagicMock
    original_hook = sys.excepthook
    try:
        mock_ultratb = MagicMock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_ultratb.assert_called_once()
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from unittest.mock import patch, MagicMock
    original_hook = sys.excepthook
    try:
        mock_ultratb = MagicMock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_ultratb.assert_not_called()
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_skip_bdbquit():
    import sys
    from unittest.mock import patch, MagicMock
    original_hook = sys.excepthook
    try:
        mock_ultratb = MagicMock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb):
            register_ipython_excepthook()
            sys.excepthook(BdbQuit, BdbQuit(), None)
            mock_ultratb.assert_not_called()
    finally:
        sys.excepthook = original_hook

def test_register_ipython_excepthook_calls_ipython_hook():
    import sys
    from unittest.mock import patch, MagicMock
    original_hook = sys.excepthook
    try:
        mock_ultratb = MagicMock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb):
            register_ipython_excepthook()
            sys.excepthook(ValueError, ValueError("test"), None)
            mock_ultratb.assert_called_once()
    finally:
        sys.excepthook = original_hook


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception():
        pass

    try:
        exception_wrapper(handler_without_exception)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #51
#--------------------------

def test_register_ipython_excepthook_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_false():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import List, Type
    class BdbQuit(BaseException):
        pass
    skip_exceptions: List[Type[BaseException]] = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    type_exc = KeyboardInterrupt
    value_exc = KeyboardInterrupt()
    traceback_exc = Mock(spec=TracebackType)
    result = any(type_exc is exc_type for exc_type in skip_exceptions)
    assert result == False


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def dummy_function():
        pass

    try:
        dummy_function()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #54
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    captured_exception = None
    captured_args = {}
    def custom_handler(e, arg1, arg2, optional_arg="default"):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "optional_arg": optional_arg}
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, optional_arg="default"):
        raise RuntimeError("custom error")
    func_with_args("value1", "value2")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["optional_arg"] == "default"

def test_exception_wrapper_with_kwargs():
    captured_kwargs = {}
    def custom_handler(e, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(custom_handler)
    def func_with_kwargs(a, b=2, **kwargs):
        raise TypeError("kwargs error")
    func_with_kwargs(1, c=3, d=4)
    assert captured_kwargs["a"] == 1
    assert captured_kwargs["b"] == 2
    assert captured_kwargs["c"] == 3
    assert captured_kwargs["d"] == 4

def test_exception_wrapper_with_generator():
    error_occurred = False
    def custom_handler(e):
        nonlocal error_occurred
        error_occurred = True
    @exception_wrapper(custom_handler)
    def generator_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = generator_func()
    result = list(gen)
    assert result == [1]
    assert error_occurred

def test_exception_wrapper_with_matching_args():
    captured = {}
    def custom_handler(e, x, y):
        nonlocal captured
        captured = {"e": e, "x": x, "y": y}
    @exception_wrapper(custom_handler)
    def func(x, y):
        raise Exception("match error")
    func(10, 20)
    assert isinstance(captured["e"], Exception)
    assert captured["x"] == 10
    assert captured["y"] == 20

def test_exception_wrapper_with_default_args_in_handler():
    captured = {}
    def custom_handler(e, a, b, extra="extra_default"):
        nonlocal captured
        captured = {"a": a, "b": b, "extra": extra}
    @exception_wrapper(custom_handler)
    def func(a, b=5):
        raise Exception("default error")
    func(100)
    assert captured["a"] == 100
    assert captured["b"] == 5
    assert captured["extra"] == "extra_default"

def test_exception_wrapper_with_var_kw_in_handler():
    captured_kw = {}
    def custom_handler(e, **kw):
        nonlocal captured_kw
        captured_kw = kw
    @exception_wrapper(custom_handler)
    def func(p, q=10):
        raise Exception("var kw error")
    func(7, q=20)
    assert captured_kw["p"] == 7
    assert captured_kw["q"] == 20

def test_exception_wrapper_with_nested_wrapped():
    captured = None
    def custom_handler(e):
        nonlocal captured
        captured = e
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper(custom_handler)
    @decorator
    def nested_func():
        raise ValueError("nested error")
    nested_func()
    assert isinstance(captured, ValueError)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func(n):
        for i in range(n):
            yield i
    result = list(gen_func(3))
    assert result == [0, 1, 2]


