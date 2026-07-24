####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_log_exception_with_user_msg():
    class TestException(Exception):
        pass
    e = TestException("test error")
    log_exception(e, user_msg="User message")

def test_log_exception_without_user_msg():
    class TestException(Exception):
        pass
    e = TestException("test error")
    log_exception(e)

def test_log_exception_with_called_process_error():
    class TestCalledProcessError(subprocess.CalledProcessError):
        pass
    e = TestCalledProcessError(1, "test command", output="test output")
    log_exception(e)

def test_log_exception_with_kwargs():
    class TestException(Exception):
        pass
    e = TestException("test error")
    log_exception(e, timestamp=False, include_proc_id=False)

def test_log_exception_with_logging_error():
    class TestException(Exception):
        pass
    e = TestException("test error")
    original_log = flutes.log.log
    flutes.log.log = lambda *args, **kwargs: 1/0
    try:
        log_exception(e)
    except Exception as log_e:
        assert isinstance(log_e, ZeroDivisionError)
    finally:
        flutes.log.log = original_log


# LLM-generated content at query #2
#--------------------------

```
def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #3
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd", output=b"output")
    log_exception(e)


# LLM-generated content at query #4
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_passes_through_return_value():
    @exception_wrapper()
    def func_that_returns():
        return 42

    result = func_that_returns()
    assert result == 42


def test_exception_wrapper_passes_through_generator():
    @exception_wrapper()
    def func_that_yields():
        yield 1
        yield 2
        yield 3

    gen = func_that_yields()
    assert list(gen) == [1, 2, 3]


def test_exception_wrapper_with_custom_handler():
    handler_called = False

    def handler(e):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()
    assert handler_called


def test_exception_wrapper_with_handler_args():
    handler_args = {}

    def handler(e, arg1, arg2, kwarg1=None):
        handler_args.update({"e": e, "arg1": arg1, "arg2": arg2, "kwarg1": kwarg1})

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two", kwarg1=3.0)
    assert isinstance(handler_args["e"], ValueError)
    assert str(handler_args["e"]) == "test error"
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == "two"
    assert handler_args["kwarg1"] == 3.0


def test_exception_wrapper_with_handler_kwargs():
    handler_kwargs = {}

    def handler(e, **kwargs):
        handler_kwargs.update({"e": e, **kwargs})

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two", kwarg1=3.0)
    assert isinstance(handler_kwargs["e"], ValueError)
    assert str(handler_kwargs["e"]) == "test error"
    assert handler_kwargs["arg1"] == 1
    assert handler_kwargs["arg2"] == "two"
    assert handler_kwargs["kwarg1"] == 3.0


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2="default"):
        assert isinstance(e, ValueError)
        assert arg1 == "value1"
        assert arg2 == "default"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2="default"):
        raise ValueError("Test error")

    test_func("value1")

def test_exception_wrapper_custom_handler_with_kwargs():
    def custom_handler(e, arg1, arg2="default", **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == "value1"
        assert arg2 == "default"
        assert kwargs == {"extra": "value"}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2="default", **kwargs):
        raise ValueError("Test error")

    test_func("value1", extra="value")

def test_exception_wrapper_generator_function():
    @exception_wrapper()
    def test_gen_func():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_exception_wrapper_custom_handler_with_mismatched_args():
    def custom_handler(e, arg1):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg2):
            pass
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_exception_wrapper_custom_handler_with_defaults():
    def custom_handler(e, arg1, arg2="default"):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1, arg2):
            pass
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #6
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        log_exception(e)


# LLM-generated content at query #7
#--------------------------

```
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    import sys
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit

    def mock_ipython_hook(type, value, traceback):
        pass

    def mock_sys_excepthook(type, value, traceback):
        assert type is KeyboardInterrupt

    original_excepthook = sys.excepthook
    sys.excepthook = mock_sys_excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    excepthook = sys.excepthook
    sys.excepthook = original_excepthook

    excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_handler_must_have_positional_argument_for_exception():
    def invalid_handler():
        pass

    try:
        exception_wrapper(invalid_handler)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper_handler_fn_validation():
    def handler_fn_without_exception_arg():
        pass

    def handler_fn_with_varargs(e, *args):
        pass

    def handler_fn_with_matching_arg(e, arg1):
        pass

    def handler_fn_with_default_arg(e, arg1=None):
        pass

    def handler_fn_with_unmatched_arg(e, arg2):
        pass

    def wrapped_fn(arg1):
        pass

    decorator = exception_wrapper(handler_fn_without_exception_arg)
    try:
        decorator(wrapped_fn)
        assert False
    except ValueError:
        pass

    decorator = exception_wrapper(handler_fn_with_varargs)
    try:
        decorator(wrapped_fn)
        assert False
    except ValueError:
        pass

    decorator = exception_wrapper(handler_fn_with_matching_arg)
    wrapped = decorator(wrapped_fn)
    assert callable(wrapped)

    decorator = exception_wrapper(handler_fn_with_default_arg)
    try:
        decorator(wrapped_fn)
        assert False
    except ValueError:
        pass

    decorator = exception_wrapper(handler_fn_with_unmatched_arg)
    try:
        decorator(wrapped_fn)
        assert False
    except ValueError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    class MockCalledProcessError:
        def __init__(self, output):
            self.output = output

    e = MockCalledProcessError(output="test output")
    log_exception(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_handler_fn_validation():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, arg3):
        pass

    func(1, 2, 3)


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        pass

    result = test_func()
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_with_custom_handler():
    def handler_fn(e):
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def func():
        yield 1
        raise ValueError("test error")

    list(func())

def test_exception_wrapper_with_custom_handler_and_args():
    def handler_fn(e, arg1, arg2):
        assert arg1 == 1
        assert arg2 == 2
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    func(1, 2)

def test_exception_wrapper_with_custom_handler_and_kwargs():
    def handler_fn(e, arg1, arg2):
        assert arg1 == 1
        assert arg2 == 2
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    func(arg1=1, arg2=2)

def test_exception_wrapper_with_custom_handler_and_mixed_args():
    def handler_fn(e, arg1, arg2):
        assert arg1 == 1
        assert arg2 == 2
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    func(1, arg2=2)

def test_exception_wrapper_with_custom_handler_and_unmatched_args():
    def handler_fn(e, arg1, arg2):
        assert arg1 == 1
        assert arg2 == 2
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, arg3):
        raise ValueError("test error")

    func(1, 2, 3)

def test_exception_wrapper_with_custom_handler_and_unmatched_kwargs():
    def handler_fn(e, arg1, arg2):
        assert arg1 == 1
        assert arg2 == 2
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, arg3):
        raise ValueError("test error")

    func(arg1=1, arg2=2, arg3=3)


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def func():
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_log_exception_with_non_CalledProcessError():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)


# LLM-generated content at query #16
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    handler_called = False
    def handler(e):
        nonlocal handler_called
        handler_called = True
    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert handler_called

def test_exception_wrapper_with_args():
    handler_args = None
    def handler(e, x, y):
        nonlocal handler_args
        handler_args = (x, y)
    @exception_wrapper(handler)
    def func_with_args(x, y):
        raise ValueError("test error")
    func_with_args(1, 2)
    assert handler_args == (1, 2)

def test_exception_wrapper_with_kwargs():
    handler_kwargs = None
    def handler(e, x, y):
        nonlocal handler_kwargs
        handler_kwargs = (x, y)
    @exception_wrapper(handler)
    def func_with_kwargs(x, y=2):
        raise ValueError("test error")
    func_with_kwargs(1)
    assert handler_kwargs == (1, 2)

def test_exception_wrapper_with_var_kwargs():
    handler_kwargs = None
    def handler(e, x, **kwargs):
        nonlocal handler_kwargs
        handler_kwargs = (x, kwargs)
    @exception_wrapper(handler)
    def func_with_var_kwargs(x, y=2, **kwargs):
        raise ValueError("test error")
    func_with_var_kwargs(1, z=3)
    assert handler_kwargs == (1, {'y': 2, 'z': 3})

def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")
    list(gen_func())

def test_exception_wrapper_invalid_handler():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_missing_arg():
    def handler(e, x):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_default_arg_conflict():
    def handler(e, x=1):
        pass
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False
    except ValueError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass
    assert True  # If no exception is raised, the test passes


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")

    faulty_function()

def test_exception_wrapper_with_custom_handler():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "test"

    @exception_wrapper(handler_fn)
    def faulty_function(arg1, arg2):
        raise ValueError("Test error")

    faulty_function(1, "test")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def faulty_generator():
        yield 1
        raise ValueError("Test error")

    gen = faulty_generator()
    assert next(gen) == 1
    next(gen)  # This should raise the exception

def test_exception_wrapper_with_matching_args():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "test"

    @exception_wrapper(handler_fn)
    def faulty_function(arg1, arg2):
        raise ValueError("Test error")

    faulty_function(1, "test")

def test_exception_wrapper_with_default_args():
    def handler_fn(e, arg1, arg2="default"):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "test"

    @exception_wrapper(handler_fn)
    def faulty_function(arg1, arg2="default"):
        raise ValueError("Test error")

    faulty_function(1, "test")

def test_exception_wrapper_with_kwargs():
    def handler_fn(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": "test"}

    @exception_wrapper(handler_fn)
    def faulty_function(arg1, **kwargs):
        raise ValueError("Test error")

    faulty_function(1, arg2="test")


# LLM-generated content at query #19
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_custom_handler():
    def handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == "test"

    @exception_wrapper(handler)
    def func_that_raises(arg1):
        raise ValueError("test error")

    func_that_raises("test")


def test_exception_wrapper_with_args_and_kwargs():
    def handler(e, arg1, kwarg1=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == "test"
        assert kwarg1 == "kwarg"
        assert kwargs == {"extra": "value"}

    @exception_wrapper(handler)
    def func_that_raises(arg1, *args, kwarg1=None, **kwargs):
        raise ValueError("test error")

    func_that_raises("test", "ignored", kwarg1="kwarg", extra="value")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass


def test_exception_wrapper_invalid_handler():
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def dummy_func():
            pass
    except ValueError:
        pass


def test_exception_wrapper_mismatched_args():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def dummy_func():
            pass
    except ValueError:
        pass


def test_exception_wrapper_with_default_values():
    def handler(e, arg1, arg2="default"):
        assert arg1 == "test"
        assert arg2 == "default"

    @exception_wrapper(handler)
    def func_that_raises(arg1):
        raise ValueError("test error")

    func_that_raises("test")


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_with_user_msg():
    exc = ValueError("invalid value")
    user_msg = "Custom error message"
    log_exception(exc, user_msg=user_msg)

def test_log_exception_without_user_msg():
    exc = TypeError("unsupported type")
    log_exception(exc)

def test_log_exception_with_additional_kwargs():
    exc = RuntimeError("runtime error")
    log_exception(exc, force_console=True, timestamp=False)

def test_log_exception_with_called_process_error():
    exc = subprocess.CalledProcessError(1, "cmd", output="output")
    log_exception(exc)

def test_log_exception_with_logging_exception():
    exc = ValueError("invalid value")
    log_exception(exc, level="invalid_level")


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_raises_value_error_for_empty_handler_argspec():
    def empty_handler_fn():
        pass

    try:
        @exception_wrapper(empty_handler_fn)
        def dummy_fn():
            pass
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #22
#--------------------------

```python
def test_register_ipython_excepthook_default():
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook
    sys.excepthook = original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #23
#--------------------------

```
def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass

    assert dummy_func.__wrapped__ is dummy_func


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_raises_error_for_handler_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def foo():
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    class MockCalledProcessError:
        def __init__(self, output):
            self.output = output

    e = MockCalledProcessError(output="some output")
    log_exception(e)


# LLM-generated content at query #27
#--------------------------

```
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    import sys
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    from unittest.mock import Mock, patch

    original_excepthook = sys.__excepthook__
    sys.__excepthook__ = Mock()
    KeyboardInterrupt = type('KeyboardInterrupt', (BaseException,), {})

    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    excepthook = sys.excepthook
    excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    
    assert sys.__excepthook__.called


# LLM-generated content at query #28
#--------------------------

```python
def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #29
#--------------------------

```
def test_register_ipython_excepthook_captures_exceptions():
    import sys
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_skips_keyboard_interrupt():
    import sys
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_captures_keyboard_interrupt():
    import sys
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_varargs():
    def handler_without_varargs(e, arg1):
        pass

    @exception_wrapper(handler_without_varargs)
    def func(arg1):
        pass

    func("test")


# LLM-generated content at query #31
#--------------------------

```python
def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2, kwarg1=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None):
        pass

    func(1, 2, kwarg1=3)


# LLM-generated content at query #33
#--------------------------

```python
def test_register_ipython_excepthook_default_behavior():
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #34
#--------------------------

def test_exception_wrapper_handler_fn_with_varkw():
    def handler_fn(e, one, two, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    foo(1, "2", "arg1", "arg2", four=4)


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_handler_with_varargs_raises_error():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass

    try:
        dummy_func()
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2, opt_arg=None):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2, opt_arg=None):
        pass

    test_func(1, 2)


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass
    assert True  # No exception raised, meaning the predicate at line 2 evaluated to False


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_with_defaults():
    def handler_fn(e, arg_with_default=None):
        pass

    @exception_wrapper(handler_fn)
    def func():
        pass


# LLM-generated content at query #40
#--------------------------

```python
def test_log_exception_with_user_msg():
    exception = ValueError("Invalid value")
    log_exception(exception, user_msg="Custom message")

def test_log_exception_without_user_msg():
    exception = TypeError("Type mismatch")
    log_exception(exception)

def test_log_exception_with_subprocess_error():
    exception = subprocess.CalledProcessError(1, "cmd", output="error output")
    log_exception(exception)

def test_log_exception_with_additional_kwargs():
    exception = RuntimeError("Runtime issue")
    log_exception(exception, level="warning", force_console=True)

def test_log_exception_with_logging_error():
    exception = Exception("Original error")
    log_exception(exception, level="invalid_level")


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper_raises_error_when_handler_fn_has_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def dummy_function():
        pass


# LLM-generated content at query #42
#--------------------------

def test_exception_wrapper_no_handler_fn():
    @exception_wrapper()
    def dummy_func():
        pass
    assert dummy_func.__wrapped__ == dummy_func


# LLM-generated content at query #43
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_valid_handler():
    def handler_fn(e, arg1, arg2, arg3=None):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2, arg3=None):
        pass

    test_func(1, 2)


# LLM-generated content at query #45
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_with_args(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_with_args(1, "two")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")
        yield 2

    list(gen_func())


def test_exception_wrapper_with_nested_wrappers():
    def handler1(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 1

    def handler2(e, arg2):
        assert isinstance(e, ValueError)
        assert arg2 == "two"

    @exception_wrapper(handler1)
    @exception_wrapper(handler2)
    def nested_func(arg1, arg2):
        raise ValueError("test error")

    nested_func(1, "two")


def test_exception_wrapper_with_kwargs():
    def handler(e, arg1, extra_kwarg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert extra_kwarg == "extra"
        assert kwargs == {"arg2": "two", "kwarg1": None}

    @exception_wrapper(handler)
    def func_with_kwargs(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_with_kwargs(1, "two", extra_kwarg="extra")


def test_exception_wrapper_with_wrapped_function():
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    def handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 1

    @exception_wrapper(handler)
    @decorator
    def decorated_func(arg1):
        raise ValueError("test error")

    decorated_func(1)


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_handler_fn_validation():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    func(1, 2)

    def invalid_handler_fn(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_fn)
        def func2(arg1, arg2):
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    def handler_with_mismatched_arg(e, arg3):
        pass

    try:
        @exception_wrapper(handler_with_mismatched_arg)
        def func3(arg1, arg2):
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    def handler_with_default_arg(e, arg1, arg2=None):
        pass

    try:
        @exception_wrapper(handler_with_default_arg)
        def func4(arg1, arg2):
            pass
        func4(1, 2)
    except ValueError:
        assert False, "Should not raise ValueError for default args"

    def handler_with_kwargs(e, arg1, **kwargs):
        pass

    @exception_wrapper(handler_with_kwargs)
    def func5(arg1, arg2):
        pass

    func5(1, 2)


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_handler_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def func():
        pass

    func()


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")
    faulty_function()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e):
        log(f"Custom handler caught: {e}", "error")
    @exception_wrapper(custom_handler)
    def faulty_function():
        raise ValueError("Test error")
    faulty_function()

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def faulty_generator():
        yield 1
        raise ValueError("Test error")
    list(faulty_generator())

def test_exception_wrapper_with_args_and_kwargs():
    def custom_handler(e, arg1, kwarg1=None):
        log(f"Custom handler caught: {e}, arg1={arg1}, kwarg1={kwarg1}", "error")
    @exception_wrapper(custom_handler)
    def faulty_function(arg1, kwarg1=None):
        raise ValueError("Test error")
    faulty_function("arg1_value", kwarg1="kwarg1_value")

def test_exception_wrapper_with_mismatched_args():
    def custom_handler(e, arg1):
        log(f"Custom handler caught: {e}, arg1={arg1}", "error")
    @exception_wrapper(custom_handler)
    def faulty_function():
        raise ValueError("Test error")
    faulty_function()

def test_exception_wrapper_with_varargs_error():
    def custom_handler(e, *args):
        pass
    @exception_wrapper(custom_handler)
    def faulty_function():
        raise ValueError("Test error")
    faulty_function()

def test_exception_wrapper_with_default_values_error():
    def custom_handler(e, arg1, kwarg1="default"):
        pass
    @exception_wrapper(custom_handler)
    def faulty_function(arg1):
        raise ValueError("Test error")
    faulty_function("arg1_value")


# LLM-generated content at query #49
#--------------------------

```
def test_exception_wrapper_checks_handler_fn_argspec():
    def handler_fn(e, arg1, arg2, kwarg1=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, *args, kwarg1=None, **kwargs):
        pass

    func(1, 2, kwarg1=3, extra=4)


# LLM-generated content at query #50
#--------------------------

```python
def test_register_ipython_excepthook_predicate_evaluates_to_true():
    capture_keyboard_interrupt = False
    register_ipython_excepthook(capture_keyboard_interrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_custom_handler():
    def handler_fn(e):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_args():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler_fn)
    def func_that_raises(arg1, arg2):
        raise ValueError("test error")

    func_that_raises(1, "two")


def test_exception_wrapper_with_kwargs():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler_fn)
    def func_that_raises(arg1, arg2):
        raise ValueError("test error")

    func_that_raises(arg1=1, arg2="two")


def test_exception_wrapper_with_default_args():
    def handler_fn(e, arg1, arg2="default"):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "default"

    @exception_wrapper(handler_fn)
    def func_that_raises(arg1):
        raise ValueError("test error")

    func_that_raises(1)


def test_exception_wrapper_with_var_kwargs():
    def handler_fn(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"
        assert arg1 == 1
        assert kwargs == {"arg2": "two", "arg3": 3}

    @exception_wrapper(handler_fn)
    def func_that_raises(arg1, **kwargs):
        raise ValueError("test error")

    func_that_raises(1, arg2="two", arg3=3)


def test_exception_wrapper_with_generator():
    def handler_fn(e):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func_that_raises():
        yield 1
        raise ValueError("test error")

    for _ in func_that_raises():
        pass


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_function():
        pass
    assert True  # The predicate at line 2 evaluates to True when handler_fn is None


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()

def test_exception_wrapper_custom_handler():
    def handler_fn(e, arg1):
        log(f"Custom handler: {arg1}, {e}", "error")
    @exception_wrapper(handler_fn)
    def func(arg1):
        raise ValueError("test error")
    func("arg1")

def test_exception_wrapper_with_args_and_kwargs():
    def handler_fn(e, arg1, arg2, kwarg1=None):
        log(f"Custom handler: {arg1}, {arg2}, {kwarg1}, {e}", "error")
    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None):
        raise ValueError("test error")
    func("arg1", "arg2", kwarg1="kwarg1")

def test_exception_wrapper_with_unmatched_handler_args():
    def handler_fn(e, arg1):
        log(f"Custom handler: {arg1}, {e}", "error")
    @exception_wrapper(handler_fn)
    def func(arg2):
        raise ValueError("test error")
    func("arg2")

def test_exception_wrapper_with_default_values():
    def handler_fn(e, arg1, arg2, kwarg1=None):
        log(f"Custom handler: {arg1}, {arg2}, {kwarg1}, {e}", "error")
    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None):
        raise ValueError("test error")
    func("arg1", "arg2")

def test_exception_wrapper_with_generator():
    def handler_fn(e, arg1):
        log(f"Custom handler: {arg1}, {e}", "error")
    @exception_wrapper(handler_fn)
    def func(arg1):
        yield arg1
        raise ValueError("test error")
    list(func("arg1"))


# LLM-generated content at query #54
#--------------------------

def test_exception_wrapper_with_empty_handler():
    def empty_handler():
        pass

    try:
        @exception_wrapper(empty_handler)
        def foo():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_varargs():
    def handler_fn(e, arg1):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1):
        pass

    func(1)


# LLM-generated content at query #56
#--------------------------

def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass
    assert dummy_func.__wrapped__ is dummy_func


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("Test error")

    func()

def test_exception_wrapper_with_custom_handler():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("Test error")

    func(1, "two")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def func():
        yield from range(3)
        raise ValueError("Test error")

    list(func())

def test_exception_wrapper_with_custom_handler_and_default_args():
    def handler_fn(e, arg1, arg2, optional_arg="default"):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert optional_arg == "default"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("Test error")

    func(1, "two")

def test_exception_wrapper_with_custom_handler_and_kwargs():
    def handler_fn(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwargs["optional_arg"] == "value"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    func(1, "two", optional_arg="value")


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_handler_fn_has_exception_argument():
    def handler_fn(e, arg1):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1):
        pass

    func(1)


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_wrapper_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def dummy_function():
        pass


# LLM-generated content at query #60
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    type_to_check = KeyboardInterrupt
    predicate_result = any(type_to_check is exc_type for exc_type in skip_exceptions)
    assert predicate_result is False


# LLM-generated content at query #61
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2, kwarg1=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None):
        pass

    func(1, 2, kwarg1=3)


# LLM-generated content at query #62
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_with_args(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_with_args(1, "two")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")

    list(gen_func())


def test_exception_wrapper_with_mismatched_args():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_with_varargs_in_handler():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_with_default_args_in_handler():
    def handler(e, arg1, arg2="default"):
        pass

    @exception_wrapper(handler)
    def func(arg1):
        raise ValueError("test error")

    func(1)


# LLM-generated content at query #63
#--------------------------

```python
def test_exception_wrapper_handler_fn_must_have_exception_arg():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #64
#--------------------------

```python
def test_exception_wrapper_handler_with_varkw():
    def handler_fn(e, my_arg=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    foo(1, "2", "arg1", "arg2", four=4)


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_without_positional_arg():
    def handler_without_positional_arg():
        pass

    @exception_wrapper(handler_without_positional_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #66
#--------------------------

```python
def test_register_ipython_excepthook_default():
    sys_excepthook_before = sys.excepthook
    register_ipython_excepthook()
    sys_excepthook_after = sys.excepthook
    assert sys_excepthook_before != sys_excepthook_after

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    sys_excepthook_before = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    sys_excepthook_after = sys.excepthook
    assert sys_excepthook_before != sys_excepthook_after

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    sys_excepthook_before = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    sys_excepthook_after = sys.excepthook
    assert sys_excepthook_before != sys_excepthook_after


# LLM-generated content at query #67
#--------------------------

```python
def test_register_ipython_excepthook_default_behavior():
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #68
#--------------------------

```
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)
    sys.excepthook = original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert not any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)
    sys.excepthook = original_excepthook


# LLM-generated content at query #69
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_without_positional_arg():
    def handler_without_positional_arg():
        pass

    try:
        exception_wrapper(handler_without_positional_arg)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_handler_without_varargs():
    def handler_with_varargs(e, *args):
        pass

    exception_wrapper(handler_with_varargs)


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    try:
        func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #72
#--------------------------

```python
def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions

def test_register_ipython_excepthook_do_not_skip_exceptions():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_no_args():
    def handler_fn(e):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass

    assert foo() is None


# LLM-generated content at query #74
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")
    
    faulty_function()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, kwarg1=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None
        assert kwargs == {"kwarg2": True}
    
    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2, *args, kwarg1=None, kwarg2=False, **kwargs):
        raise ValueError("Test error")
    
    faulty_function(1, "two", kwarg2=True)

def test_exception_wrapper_generator_function():
    def custom_handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 1
    
    @exception_wrapper(custom_handler)
    def faulty_generator(arg1):
        yield 1
        raise ValueError("Test error")
    
    list(faulty_generator(1))

def test_exception_wrapper_without_matching_arg():
    def custom_handler(e, non_existent_arg):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def faulty_function():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_with_default_value():
    def custom_handler(e, arg1, arg2="default"):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def faulty_function(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #75
#--------------------------

Here are the test cases for the `exception_wrapper` function:


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_with_user_msg():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")

def test_log_exception_without_user_msg():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e)

def test_log_exception_with_called_process_error():
    try:
        raise subprocess.CalledProcessError(1, "cmd", output="Test output")
    except subprocess.CalledProcessError as e:
        log_exception(e)

def test_log_exception_with_additional_kwargs():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_exception(e, force_console=True)

def test_log_exception_with_logging_failure():
    try:
        raise ValueError("Test error")
    except ValueError as e:
        try:
            log_exception(e, level="invalid_level")
        except ValueError:
            pass


# LLM-generated content at query #2
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")

    faulty_function()


def test_exception_wrapper_with_custom_handler():
    def custom_handler(e):
        assert isinstance(e, ValueError)
        assert str(e) == "Test error"

    @exception_wrapper(custom_handler)
    def faulty_function():
        raise ValueError("Test error")

    faulty_function()


def test_exception_wrapper_with_handler_args():
    def custom_handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert str(e) == "Test error"
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 == "three"

    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2, kwarg1=None):
        raise ValueError("Test error")

    faulty_function(1, "two", kwarg1="three")


def test_exception_wrapper_with_handler_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert str(e) == "Test error"
        assert arg1 == 1
        assert arg2 == "two"
        assert kwargs == {"kwarg1": "three"}

    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2, kwarg1=None):
        raise ValueError("Test error")

    faulty_function(1, "two", kwarg1="three")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def faulty_generator():
        yield 1
        raise ValueError("Test error")

    gen = faulty_generator()
    next(gen)
    try:
        next(gen)
    except ValueError:
        pass
    else:
        assert False, "Exception not raised in generator"


def test_exception_wrapper_with_nested_decorator():
    def custom_handler(e):
        assert isinstance(e, ValueError)
        assert str(e) == "Test error"

    @exception_wrapper(custom_handler)
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")

    faulty_function()


# LLM-generated content at query #3
#--------------------------

Here are the test cases for the `exception_wrapper` function:


# LLM-generated content at query #4
#--------------------------

def test_log_exception_with_non_called_process_error():
    class TestException(Exception):
        pass

    e = TestException("test exception")
    log_exception(e)

def test_log_exception_with_called_process_error_no_output():
    e = subprocess.CalledProcessError(1, "cmd")
    log_exception(e)

def test_log_exception_with_called_process_error_with_output():
    e = subprocess.CalledProcessError(1, "cmd", output=b"output")
    log_exception(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_raises_error_when_handler_fn_has_varargs():
    def handler_fn_with_varargs(e, *args):
        pass

    exception_wrapper(handler_fn_with_varargs)


# LLM-generated content at query #6
#--------------------------

```python
def test_register_ipython_excepthook_with_capture_keyboard_interrupt():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_without_capture_keyboard_interrupt():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #7
#--------------------------

```python
def test_register_ipython_excepthook_does_not_capture_keyboard_interrupt():
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    else:
        assert False, "KeyboardInterrupt should not be captured"

def test_register_ipython_excepthook_captures_other_exceptions():
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        raise ValueError
    except ValueError:
        pass
    else:
        assert False, "ValueError should be captured"


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper_basic():
    @exception_wrapper()
    def divide(a, b):
        return a / b

    divide(4, 2)
    try:
        divide(4, 0)
    except ZeroDivisionError:
        pass


def test_exception_wrapper_custom_handler():
    def handler(e, a, b):
        assert isinstance(e, ZeroDivisionError)
        assert a == 4
        assert b == 0

    @exception_wrapper(handler)
    def divide(a, b):
        return a / b

    divide(4, 2)
    try:
        divide(4, 0)
    except ZeroDivisionError:
        pass


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_numbers(n):
        for i in range(n):
            yield i / (n - i - 1)

    list(gen_numbers(1))
    try:
        list(gen_numbers(2))
    except ZeroDivisionError:
        pass


def test_exception_wrapper_kwargs():
    def handler(e, a, b, c=3):
        assert isinstance(e, ZeroDivisionError)
        assert a == 4
        assert b == 0
        assert c == 3

    @exception_wrapper(handler)
    def divide(a, b, c=3):
        return a / b

    divide(4, 2)
    try:
        divide(4, 0)
    except ZeroDivisionError:
        pass


def test_exception_wrapper_var_kwargs():
    def handler(e, a, b, **kwargs):
        assert isinstance(e, ZeroDivisionError)
        assert a == 4
        assert b == 0
        assert kwargs == {'c': 3, 'd': 4}

    @exception_wrapper(handler)
    def divide(a, b, c=3, **kwargs):
        return a / b

    divide(4, 2, d=4)
    try:
        divide(4, 0, d=4)
    except ZeroDivisionError:
        pass


# LLM-generated content at query #9
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two")


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    list(gen_that_raises())


def test_exception_wrapper_nested():
    def handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 1

    @exception_wrapper(handler)
    def outer_func(arg1):
        @exception_wrapper()
        def inner_func():
            raise ValueError("test error")
        inner_func()

    outer_func(1)


def test_exception_wrapper_kwargs():
    def handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": "two", "kwarg1": None}

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two")


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    class DummyException(Exception):
        pass

    e = DummyException("test exception")
    log_exception(e)

def test_log_exception_with_called_process_error_no_output():
    e = subprocess.CalledProcessError(1, "cmd")
    log_exception(e)

def test_log_exception_with_called_process_error_with_output():
    e = subprocess.CalledProcessError(1, "cmd", output=b"output")
    log_exception(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_handler_with_varargs():
    def handler(e, *args):
        pass

    @exception_wrapper(handler)
    def func():
        pass

    exception_wrapper(handler)


# LLM-generated content at query #12
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    e = Exception("Test exception")
    log_exception(e)

def test_log_exception_with_called_process_error_and_no_output():
    e = subprocess.CalledProcessError(1, "test")
    log_exception(e)

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "test", output="test output")
    log_exception(e)


# LLM-generated content at query #13
#--------------------------

```
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass

    assert dummy_func() is None

def test_exception_wrapper_with_handler_no_args():
    def handler(e):
        pass

    @exception_wrapper(handler)
    def dummy_func():
        pass

    assert dummy_func() is None

def test_exception_wrapper_with_handler_matching_args():
    def handler(e, arg1):
        pass

    @exception_wrapper(handler)
    def dummy_func(arg1):
        pass

    assert dummy_func(1) is None

def test_exception_wrapper_with_handler_default_args():
    def handler(e, arg1=1):
        pass

    @exception_wrapper(handler)
    def dummy_func():
        pass

    assert dummy_func() is None

def test_exception_wrapper_with_handler_varargs():
    try:
        def handler(e, *args):
            pass

        @exception_wrapper(handler)
        def dummy_func():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"
    else:
        assert False, "Expected ValueError"

def test_exception_wrapper_with_handler_no_exception_arg():
    try:
        def handler():
            pass

        @exception_wrapper(handler)
        def dummy_func():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"
    else:
        assert False, "Expected ValueError"

def test_exception_wrapper_with_handler_mismatched_args():
    try:
        def handler(e, arg1):
            pass

        @exception_wrapper(handler)
        def dummy_func():
            pass
    except ValueError as e:
        assert f"Argument 'arg1' in exception handler does not match any argument in wrapped method" in str(e)
    else:
        assert False, "Expected ValueError"

def test_exception_wrapper_with_handler_conflicting_default_args():
    try:
        def handler(e, arg1=1):
            pass

        @exception_wrapper(handler)
        def dummy_func(arg1):
            pass
    except ValueError as e:
        assert f"Argument 'arg1' matches wrapped method argument, thus cannot have default values" in str(e)
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #14
#--------------------------

```python
def test_log_exception_with_CalledProcessError_and_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output="test_output")
    log_exception(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_custom_handler():
    handler_called = False

    def handler(e):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()
    assert handler_called


def test_exception_wrapper_with_args():
    handler_args = None

    def handler(e, arg1, arg2):
        nonlocal handler_args
        handler_args = (arg1, arg2)
        assert isinstance(e, ValueError)

    @exception_wrapper(handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    func_with_args(1, "two")
    assert handler_args == (1, "two")


def test_exception_wrapper_with_kwargs():
    handler_kwargs = None

    def handler(e, kw1=None, **kwargs):
        nonlocal handler_kwargs
        handler_kwargs = kwargs
        assert kw1 == "default"
        assert isinstance(e, ValueError)

    @exception_wrapper(handler)
    def func_with_kwargs(arg1, **kwargs):
        raise ValueError("test error")

    func_with_kwargs(1, extra="value")
    assert handler_kwargs == {"arg1": 1, "extra": "value"}


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = gen_func()
    assert next(gen) == 1
    next(gen)  # should raise but be caught by handler


def test_exception_wrapper_invalid_handler():
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_nested_wrapping():
    def handler(e):
        assert isinstance(e, ValueError)

    @exception_wrapper(handler)
    @exception_wrapper(handler)
    def nested_func():
        raise ValueError("test error")

    nested_func()


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_handler_with_varargs_raises_value_error():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def func():
        pass

    try:
        func()
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    predicate = any(type is exc_type for exc_type in skip_exceptions)
    skip_exceptions = [BdbQuit, KeyboardInterrupt]
    type = KeyboardInterrupt
    assert predicate == True


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_handler_without_varargs():
    def handler_fn(e):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass

    foo()


# LLM-generated content at query #19
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    exception = ValueError("test error")
    log_exception(exception)


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def faulty_func():
        raise ValueError("Test error")

    try:
        faulty_func()
    except ValueError:
        pass

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1):
        assert str(e) == "Test error"
        assert arg1 == "test_arg"

    @exception_wrapper(custom_handler)
    def faulty_func(arg1):
        raise ValueError("Test error")

    try:
        faulty_func("test_arg")
    except ValueError:
        pass

def test_exception_wrapper_custom_handler_with_kwargs():
    def custom_handler(e, arg1, kwarg1=None):
        assert str(e) == "Test error"
        assert arg1 == "test_arg"
        assert kwarg1 == "test_kwarg"

    @exception_wrapper(custom_handler)
    def faulty_func(arg1, kwarg1=None):
        raise ValueError("Test error")

    try:
        faulty_func("test_arg", kwarg1="test_kwarg")
    except ValueError:
        pass

def test_exception_wrapper_generator_function():
    @exception_wrapper()
    def faulty_gen():
        yield 1
        raise ValueError("Test error")

    gen = faulty_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass

def test_exception_wrapper_custom_handler_with_generator():
    def custom_handler(e, arg1):
        assert str(e) == "Test error"
        assert arg1 == "test_arg"

    @exception_wrapper(custom_handler)
    def faulty_gen(arg1):
        yield 1
        raise ValueError("Test error")

    gen = faulty_gen("test_arg")
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass


# LLM-generated content at query #21
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two")


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    list(gen_that_raises())


def test_exception_wrapper_custom_handler_with_kwargs():
    def handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwargs == {"kwarg1": 3, "kwarg2": 4}

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None, **kwargs):
        raise ValueError("test error")

    func_that_raises(1, "two", kwarg1=3, kwarg2=4)


def test_exception_wrapper_invalid_handler_no_args():
    try:
        @exception_wrapper(lambda: None)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_varargs():
    try:
        @exception_wrapper(lambda e, *args: None)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_mismatched_arg():
    try:
        @exception_wrapper(lambda e, non_existent_arg: None)
        def dummy(arg):
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_default_value():
    try:
        @exception_wrapper(lambda e, arg=1: None)
        def dummy(arg):
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #23
#--------------------------

```python
def test_log_exception_predicate_false():
    class MockCalledProcessError:
        output = "some output"

    e = MockCalledProcessError()
    log_exception(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_raises_value_error_for_handler_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def foo():
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    result = register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert result is None


# LLM-generated content at query #26
#--------------------------

```python
def test_register_ipython_excepthook_default():
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_excepthook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_excepthook


# LLM-generated content at query #27
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs():
    def handler_fn(e, *args):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_with_varargs_in_handler():
    def handler_fn(e, *args):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_handler_must_have_positional_argument_for_exception():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def foo():
        pass

    try:
        foo()
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_handler_without_varargs():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    func(1, 2)


# LLM-generated content at query #31
#--------------------------

```
def test_exception_wrapper_handler_fn_with_varkw():
    def handler_fn(e, one, two, my_arg=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    foo(1, "2", "arg1", "arg2", four=4)


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass
    assert True


# LLM-generated content at query #33
#--------------------------

```python
def test_register_ipython_excepthook_skip_exceptions():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #34
#--------------------------

```
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    assert KeyboardInterrupt not in skip_exceptions

def test_register_ipython_excepthook_include_keyboard_interrupt():
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_with_invalid_handler():
    def invalid_handler():
        pass

    @exception_wrapper(invalid_handler)
    def dummy_function():
        pass

    try:
        dummy_function()
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_handler_fn_no_varargs():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def foo(arg1, arg2):
        pass

    foo(1, 2)


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_custom_handler():
    def handler_fn(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 42

    @exception_wrapper(handler_fn)
    def func(arg1):
        raise ValueError("test error")

    func(42)

def test_exception_wrapper_with_kwargs():
    def handler_fn(e, arg1, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 42
        assert kwarg1 == "test"

    @exception_wrapper(handler_fn)
    def func(arg1, kwarg1=None):
        raise ValueError("test error")

    func(42, kwarg1="test")

def test_exception_wrapper_with_args_and_kwargs():
    def handler_fn(e, arg1, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 42
        assert kwarg1 == "test"

    @exception_wrapper(handler_fn)
    def func(arg1, kwarg1=None):
        raise ValueError("test error")

    func(42, kwarg1="test")

def test_exception_wrapper_with_generator():
    def handler_fn(e):
        assert isinstance(e, ValueError)

    @exception_wrapper(handler_fn)
    def func():
        yield 42
        raise ValueError("test error")

    list(func())

def test_exception_wrapper_with_nested_function():
    def handler_fn(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == 42

    @exception_wrapper(handler_fn)
    def outer_func(arg1):
        def inner_func():
            raise ValueError("test error")
        inner_func()

    outer_func(42)


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_arg():
    def handler_without_exception():
        pass

    @exception_wrapper(handler_without_exception)
    def dummy_func():
        pass

    try:
        dummy_func()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #39
#--------------------------

```python
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def dummy_func():
        pass
    assert True  # If no exception is raised, the test passes


# LLM-generated content at query #41
#--------------------------

```
def test_register_ipython_excepthook_skip_keyboard_interrupt():
    import sys
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    from unittest.mock import MagicMock

    original_excepthook = sys.__excepthook__
    sys.__excepthook__ = MagicMock()
    KeyboardInterrupt = type('KeyboardInterrupt', (BaseException,), {})
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    excepthook = sys.excepthook
    excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    sys.__excepthook__.assert_called_once()


# LLM-generated content at query #42
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd", output="some output")
    log_exception(e)


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    func()


def test_exception_wrapper_custom_handler():
    def handler_fn(e, one, two, my_arg=None):
        assert isinstance(e, ValueError)
        assert e.args[0] == "test error"
        assert one == 1
        assert two == "2"
        assert my_arg is None

    @exception_wrapper(handler_fn)
    def func(one, two):
        raise ValueError("test error")

    func(1, "2")


def test_exception_wrapper_custom_handler_with_kwargs():
    def handler_fn(e, one, two, my_arg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert e.args[0] == "test error"
        assert one == 1
        assert two == "2"
        assert my_arg is None
        assert kwargs == {"three": 3}

    @exception_wrapper(handler_fn)
    def func(one, two, **kwargs):
        raise ValueError("test error")

    func(1, "2", three=3)


def test_exception_wrapper_generator():
    @exception_wrapper()
    def func():
        yield 1
        raise ValueError("test error")

    result = func()
    assert next(result) == 1
    try:
        next(result)
    except StopIteration:
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_skip_exceptions_contains_keyboard_interrupt():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def faulty_function():
        raise ValueError("Test error")

    faulty_function()


def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == "test_arg1"
        assert arg2 == "test_arg2"

    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2):
        raise ValueError("Test error")

    faulty_function("test_arg1", "test_arg2")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def faulty_generator():
        yield 1
        raise ValueError("Test error")

    gen = faulty_generator()
    next(gen)
    next(gen)


def test_exception_wrapper_with_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == "test_arg1"
        assert arg2 == "test_arg2"
        assert kwargs["extra_arg"] == "extra_value"

    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    faulty_function("test_arg1", "test_arg2", extra_arg="extra_value")


def test_exception_wrapper_with_default_args():
    def custom_handler(e, arg1, arg2, default_arg="default"):
        assert isinstance(e, ValueError)
        assert arg1 == "test_arg1"
        assert arg2 == "test_arg2"
        assert default_arg == "default"

    @exception_wrapper(custom_handler)
    def faulty_function(arg1, arg2, default_arg="default"):
        raise ValueError("Test error")

    faulty_function("test_arg1", "test_arg2")


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_handler_fn_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def foo():
        pass

    foo()


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_without_exception_arg():
    def handler_fn():
        pass

    try:
        @exception_wrapper(handler_fn)
        def func():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def func():
        pass


# LLM-generated content at query #49
#--------------------------

```
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_custom_handler():
    def handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == "test_arg"

    @exception_wrapper(handler)
    def func_that_raises(arg1):
        raise ValueError("test error")

    func_that_raises("test_arg")


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_that_raises():
        yield 1
        raise ValueError("test error")

    list(gen_that_raises())


def test_exception_wrapper_custom_handler_with_kwargs():
    def handler(e, arg1, extra_arg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == "test_arg"
        assert extra_arg == "extra"
        assert kwargs == {"kwarg1": "kw1"}

    @exception_wrapper(handler)
    def func_that_raises(arg1, kwarg1=None):
        raise ValueError("test error")

    func_that_raises("test_arg", kwarg1="kw1", extra_arg="extra")


def test_exception_wrapper_invalid_handler_no_args():
    def handler():
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_varargs():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_missing_arg():
    def handler(e, missing_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_exception_wrapper_invalid_handler_default_value():
    def handler(e, arg1="default"):
        pass

    try:
        @exception_wrapper(handler)
        def func(arg1):
            pass
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# LLM-generated content at query #50
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_that_raises():
        yield 1
        raise ValueError("test error")

    list(gen_that_raises())


def test_exception_wrapper_with_nested_args():
    def handler(e, arg1, arg2, kwarg1=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None
        assert kwargs == {"kwarg2": 2}

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None, **kwargs):
        raise ValueError("test error")

    func_that_raises(1, "two", kwarg2=2)


def test_exception_wrapper_with_mismatched_args():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError:
        pass
    else:
        assert False, "Should raise ValueError for mismatched args"


def test_exception_wrapper_with_varargs():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError:
        pass
    else:
        assert False, "Should raise ValueError for varargs in handler"


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper_with_no_handler_fn():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_with_custom_handler_fn():
    def handler_fn(e):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_with_custom_handler_fn_and_args():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    func(1, "two")

def test_exception_wrapper_with_generator():
    def handler_fn(e):
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(handler_fn)
    def func():
        yield 1
        raise ValueError("test error")

    list(func())


# LLM-generated content at query #52
#--------------------------

```
def test_register_ipython_excepthook_skip_exceptions():
    skip_exceptions = [BdbQuit]
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_handler_fn_must_have_positional_arg_for_exception():
    def invalid_handler():
        pass

    @exception_wrapper(invalid_handler)
    def dummy_func():
        pass


# LLM-generated content at query #54
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    func()

def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2, custom_arg=None):
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "two"
        assert custom_arg is None

    @exception_wrapper(handler)
    def func(arg1, arg2, *args, custom_arg=None, **kwargs):
        raise ValueError("test error")

    func(1, "two", custom_arg=None)

def test_exception_wrapper_with_generator():
    def handler(e, arg1, arg2):
        assert str(e) == "test error"
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler)
    def func(arg1, arg2):
        yield
        raise ValueError("test error")

    list(func(1, "two"))

def test_exception_wrapper_with_mismatched_argument():
    def handler(e, arg1, arg2, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
    except ValueError:
        pass
    else:
        assert False

def test_exception_wrapper_with_default_value_for_matched_argument():
    def handler(e, arg1, arg2=2):
        pass

    try:
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
    except ValueError:
        pass
    else:
        assert False

def test_exception_wrapper_with_varargs():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError:
        pass
    else:
        assert False

def test_exception_wrapper_with_kwargs():
    def handler(e, **kwargs):
        pass

    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("test error")

    func(1, arg2="two")


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper_with_no_handler_fn():
    @exception_wrapper()
    def dummy_function():
        pass

    assert dummy_function() is None


# LLM-generated content at query #56
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def raises_error():
        raise ValueError("test error")

    raises_error()


def test_exception_wrapper_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def raises_error(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    raises_error(1, "two")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")

    list(gen_func())


def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"

    @exception_wrapper(handler)
    def raises_error(arg1, arg2):
        raise ValueError("test error")

    raises_error(1, "two")


def test_exception_wrapper_with_kwargs():
    def handler(e, arg1, arg2, kwarg1=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None
        assert kwargs == {"extra": "value"}

    @exception_wrapper(handler)
    def raises_error(arg1, arg2, kwarg1=None, **kwargs):
        raise ValueError("test error")

    raises_error(1, "two", extra="value")


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_wrapper_with_invalid_handler_arg():
    def invalid_handler():
        pass

    @exception_wrapper(invalid_handler)
    def dummy_function():
        pass

    try:
        dummy_function()
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_handler_fn_requires_positional_argument():
    def handler_without_positional_arg():
        pass

    def handler_with_positional_arg(e):
        pass

    @exception_wrapper(handler_without_positional_arg)
    def func():
        pass

    try:
        func()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_handler_with_varargs_raises_value_error():
    def handler_fn(e, *args):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass

    foo()


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_handler_must_have_positional_argument_for_exception():
    def handler_fn():
        pass

    try:
        @exception_wrapper(handler_fn)
        def foo():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #61
#--------------------------

```
def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook


# LLM-generated content at query #62
#--------------------------

```python
def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, optional_arg=None):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        pass

    test_func(1, 2)


# LLM-generated content at query #63
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def foo():
        pass


# LLM-generated content at query #64
#--------------------------

```python
def test_exception_wrapper_handler_fn_requires_exception_argument():
    def handler_fn_without_exception_arg():
        pass

    @exception_wrapper(handler_fn_without_exception_arg)
    def dummy_function():
        pass

    try:
        dummy_function()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("Test error")

    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert str(e) == "Test error"
        assert arg1 == 1
        assert arg2 == "test"

    @exception_wrapper(custom_handler)
    def func_that_raises(arg1, arg2):
        raise ValueError("Test error")

    func_that_raises(1, "test")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def func_with_generator():
        yield 1
        raise ValueError("Test error")

    gen = func_with_generator()
    assert next(gen) == 1
    next(gen)

def test_exception_wrapper_with_mismatched_args():
    def custom_handler(e, arg1, arg2):
        pass

    try:
        @exception_wrapper(custom_handler)
        def func_with_mismatched_args(arg1):
            pass
    except ValueError:
        pass

def test_exception_wrapper_with_default_args():
    def custom_handler(e, arg1, arg2="default"):
        assert str(e) == "Test error"
        assert arg1 == 1
        assert arg2 == "default"

    @exception_wrapper(custom_handler)
    def func_with_default_args(arg1):
        raise ValueError("Test error")

    func_with_default_args(1)


# LLM-generated content at query #66
#--------------------------

```python
def test_register_ipython_excepthook_with_capture_keyboard_interrupt_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions

def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #67
#--------------------------

```python
def test_exception_wrapper_handler_fn_has_varargs():
    def handler_with_varargs(e, *args):
        pass

    @exception_wrapper(handler_with_varargs)
    def dummy_function():
        pass


# LLM-generated content at query #68
#--------------------------

```python
def test_register_ipython_excepthook_skip_exceptions():
    skip_exceptions = [BdbQuit]
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #69
#--------------------------

```python
def test_handler_fn_must_have_exception_argument():
    def handler_fn_without_args():
        pass

    @exception_wrapper(handler_fn_without_args)
    def foo():
        pass

    try:
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper_handler_fn_with_varkw():
    def handler_fn(e, one, two, my_arg=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, three=None, **kwargs):
        pass

    foo(1, "2", three=3, four=4)


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_no_handler():
    def dummy_func():
        pass

    wrapped_func = exception_wrapper()(dummy_func)
    result = wrapped_func()
    assert result is None


# LLM-generated content at query #72
#--------------------------

```python
def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook != original_hook
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook != original_hook
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #73
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()


def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 is None

    @exception_wrapper(handler)
    def func_that_raises(arg1, arg2, kwarg1=None):
        raise ValueError("test error")

    func_that_raises(1, "two")


def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    list(gen_that_raises())


def test_exception_wrapper_with_nested_args():
    def handler(e, arg1, arg2, kwarg1=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "two"
        assert kwarg1 == "three"

    @exception_wrapper(handler)
    def func_that_raises(arg1, *args, arg2=None, **kwargs):
        raise ValueError("test error")

    func_that_raises(1, "extra", arg2="two", kwarg1="three")


def test_exception_wrapper_with_invalid_handler():
    def invalid_handler(*args):
        pass

    try:
        @exception_wrapper(invalid_handler)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_with_missing_arg():
    def handler(e, missing_arg):
        pass

    try:
        @exception_wrapper(handler)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exception_wrapper_with_default_arg_conflict():
    def handler(e, arg1, arg2="default"):
        pass

    try:
        @exception_wrapper(handler)
        def dummy_func(arg1, arg2):
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


