####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
        assert True

def test_log_exception_called_process_error_with_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
        assert True

def test_log_exception_called_process_error_without_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=None)
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
    finally:
        log = original_log

def test_log_exception_with_kwargs():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)
        assert True


# LLM-generated content at query #2
#--------------------------

```python
def test_log_exception_does_not_log_traceback_for_called_process_error_with_output():
    import subprocess
    import flutes.exception
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"error output")
    flutes.exception.log_exception(e)


# LLM-generated content at query #3
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = "output"
    flutes.exception.log_exception(e)


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    handler_called = []
    def custom_handler(e):
        handler_called.append(e)
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)
    assert str(handler_called[0]) == "test error"

def test_exception_wrapper_with_matching_arguments():
    handler_args = []
    def custom_handler(e, arg1, arg2):
        handler_args.extend([arg1, arg2])
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("error")
    func_with_args("value1", "value2")
    assert handler_args == ["value1", "value2"]

def test_exception_wrapper_with_kwargs():
    handler_kwargs = {}
    def custom_handler(e, **kwargs):
        handler_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def func_with_kwargs(a, b=2):
        raise Exception("error")
    func_with_kwargs(1, b=3)
    assert handler_kwargs == {"a": 1, "b": 3}

def test_exception_wrapper_with_default_arguments():
    handler_values = []
    def custom_handler(e, x, y=10):
        handler_values.extend([x, y])
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, y=5):
        raise ValueError("error")
    func_with_defaults(1)
    assert handler_values == [1, 5]

def test_exception_wrapper_with_generator():
    handler_called = []
    def custom_handler(e):
        handler_called.append(True)
    @exception_wrapper(custom_handler)
    def generator_func():
        yield 1
        raise ValueError("generator error")
    gen = generator_func()
    result = list(gen)
    assert result == [1]
    assert handler_called == [True]

def test_exception_wrapper_with_nested_wraps():
    handler_called = []
    def custom_handler(e):
        handler_called.append(True)
    @exception_wrapper(custom_handler)
    @functools.wraps(lambda: None)
    def wrapped_func():
        raise ValueError("error")
    wrapped_func()
    assert handler_called == [True]

def test_exception_wrapper_with_mixed_arguments():
    captured_args = {}
    def custom_handler(e, a, b, c=30, **kwargs):
        captured_args.update({"a": a, "b": b, "c": c, "kwargs": kwargs})
    @exception_wrapper(custom_handler)
    def mixed_func(a, *args, b=20, **kwargs):
        raise TypeError("mixed error")
    mixed_func(1, "extra", b=2, d=4)
    assert captured_args["a"] == 1
    assert captured_args["b"] == 2
    assert captured_args["c"] == 30
    assert captured_args["kwargs"] == {"args": ("extra",), "kwargs": {"d": 4}}

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def normal_func():
        return 42
    result = normal_func()
    assert result == 42

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def normal_generator():
        yield from range(3)
    result = list(normal_generator())
    assert result == [0, 1, 2]

def test_exception_wrapper_with_handler_varargs_error():
    try:
        def invalid_handler(*args):
            pass
        @exception_wrapper(invalid_handler)
        def some_func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_handler_no_args_error():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def some_func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_unmatched_handler_arg():
    try:
        def handler(e, unmatched):
            pass
        @exception_wrapper(handler)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_with_matched_arg_has_default():
    try:
        def handler(e, x=1):
            pass
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False
    except ValueError:
        pass


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command '['ls']' returned non-zero exit status 1.", "error")
    e_no_output = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=None)
    with patch('flutes.log.log') as mock_log:
        with patch('traceback.format_exc', return_value="traceback"):
            log_exception(e_no_output)
            mock_log.assert_any_call("traceback", "error")
            mock_log.assert_any_call("<CalledProcessError> Command '['ls']' returned non-zero exit status 1.", "error")
    e_other = ValueError("test error")
    with patch('flutes.log.log') as mock_log:
        with patch('traceback.format_exc', return_value="traceback"):
            log_exception(e_other)
            mock_log.assert_any_call("traceback", "error")
            mock_log.assert_any_call("<ValueError> test error", "error")


# LLM-generated content at query #7
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output="output")
    log_exception(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass

    @exception_wrapper(handler_without_exception_arg)
    def dummy_function():
        pass


# LLM-generated content at query #9
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def raise_exception():
        raise ValueError("test error")
    raise_exception()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def raise_exception():
        raise ValueError("test error")
    raise_exception()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured = {}
    def custom_handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def raise_exception(arg1, arg2):
        raise ValueError("test error")
    raise_exception(10, "hello")
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == "hello"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def custom_handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def raise_exception(arg1, arg2=None):
        raise ValueError("test error")
    raise_exception(10, arg2="world")
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 10
    assert captured['kwargs'] == {'arg2': 'world'}

def test_exception_wrapper_custom_handler_with_default_args():
    captured = {}
    def custom_handler(e, arg1, my_default=100):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['my_default'] = my_default
    @exception_wrapper(custom_handler)
    def raise_exception(arg1):
        raise ValueError("test error")
    raise_exception(42)
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 42
    assert captured['my_default'] == 100

def test_exception_wrapper_generator():
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

def test_exception_wrapper_generator_custom_handler():
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

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def add(a, b):
        return a + b
    result = add(3, 4)
    assert result == 7

def test_exception_wrapper_no_exception_generator():
    @exception_wrapper()
    def gen_numbers(n):
        for i in range(n):
            yield i
    result = list(gen_numbers(3))
    assert result == [0, 1, 2]

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

def test_exception_wrapper_invalid_handler_missing_arg():
    try:
        def custom_handler(e, missing_arg):
            pass
        @exception_wrapper(custom_handler)
        def dummy(arg1):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_invalid_handler_default_arg_matches():
    try:
        def custom_handler(e, arg1, default_arg=10):
            pass
        @exception_wrapper(custom_handler)
        def dummy(arg1, default_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #10
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
    captured = {}
    def handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise RuntimeError("error")
    func_raises(1, arg2=2)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 1
    assert captured['arg2'] == 2

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def handler(e, arg1, arg2, extra="default"):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['extra'] = extra
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise Exception("error")
    func_raises(10, 20)
    assert captured['arg1'] == 10
    assert captured['arg2'] == 20
    assert captured['extra'] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func_raises(arg1, arg2, extra=None):
        raise Exception("error")
    func_raises(5, arg2=6, extra=7)
    assert captured['arg1'] == 5
    assert captured['kwargs'] == {'arg2': 6, 'extra': 7}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_generator_exception():
    caught = False
    def handler(e):
        nonlocal caught
        caught = True
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert caught

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

def test_exception_wrapper_missing_arg_in_wrapped():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func(arg1):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_default_arg_matches_wrapped():
    try:
        def handler(e, arg1, arg2="default"):
            pass
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_already_decorated():
    def dummy_decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper()
    @dummy_decorator
    def func():
        raise ValueError("test")
    func()


# LLM-generated content at query #11
#--------------------------

def test_log_exception_with_user_msg():
    import subprocess
    import traceback
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="User message", force_console=False)

def test_log_exception_without_user_msg():
    import subprocess
    import traceback
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e, timestamp=False)

def test_log_exception_called_process_error_with_output():
    import subprocess
    import traceback
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd="ls", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e, include_proc_id=False)

def test_log_exception_called_process_error_without_output():
    import subprocess
    import traceback
    try:
        raise subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    except subprocess.CalledProcessError as e:
        log_exception(e, level="error")

def test_log_exception_logging_failure():
    import subprocess
    import traceback
    original_log = log
    def failing_log(*args, **kwargs):
        raise RuntimeError("log failed")
    import flutes.log as log_module
    log_module.log = failing_log
    try:
        try:
            raise KeyError("key missing")
        except KeyError as e:
            log_exception(e)
    except RuntimeError as e:
        assert str(e) == "log failed"
    finally:
        log_module.log = original_log


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)(lambda: None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped_method_arg():
    def handler_fn(e, arg1, arg2=5):
        pass

    @exception_wrapper(handler_fn)
    def wrapped_func(arg1, arg2):
        pass

    try:
        wrapped_func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #16
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    captured_exception = None
    captured_args = {}
    def custom_handler(e, arg1, arg2, extra="default"):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "extra": extra}
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("custom error")
    func_with_args("value1", arg2="value2")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["extra"] == "default"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_kwargs = {}
    def custom_handler(e, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(custom_handler)
    def func_with_kwargs(a, b=2, **kwargs):
        raise TypeError("kwargs error")
    func_with_kwargs(1, c=3)
    assert captured_kwargs["a"] == 1
    assert captured_kwargs["b"] == 2
    assert captured_kwargs["c"] == 3

def test_exception_wrapper_generator():
    log_messages = []
    original_log = log
    def mock_log(msg, level="error", **kwargs):
        log_messages.append(msg)
    import flutes.exception
    flutes.exception.log = mock_log
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = generator_that_raises()
    result = list(gen)
    flutes.exception.log = original_log
    assert result == [1]
    assert any("generator error" in msg for msg in log_messages)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_handler_with_matching_args():
    captured = []
    def custom_handler(e, first, second):
        captured.extend([first, second])
    @exception_wrapper(custom_handler)
    def func(first, second):
        raise Exception("match")
    func(10, 20)
    assert captured == [10, 20]

def test_exception_wrapper_handler_with_defaults_and_kwargs():
    captured = {}
    def custom_handler(e, required, optional="opt_default", **extra):
        captured["required"] = required
        captured["optional"] = optional
        captured["extra"] = extra
    @exception_wrapper(custom_handler)
    def func(req, opt=100, **kw):
        raise Exception("test")
    func(999, extra_key="extra_value")
    assert captured["required"] == 999
    assert captured["optional"] == 100
    assert captured["extra"] == {"opt": 100, "extra_key": "extra_value"}

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_invalid_handler_missing_arg():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
        func(1)
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_invalid_handler_default_arg_matches():
    try:
        def handler(e, arg_with_default="default"):
            pass
        @exception_wrapper(handler)
        def func(arg_with_default):
            pass
        func(1)
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_log_exception_failure():
    import flutes.exception
    original_log = flutes.exception.log
    def failing_log(msg, level="error", **kwargs):
        raise RuntimeError("log failed")
    flutes.exception.log = failing_log
    @exception_wrapper()
    def failing_func():
        raise ValueError("original error")
    try:
        failing_func()
    except RuntimeError as e:
        assert "log failed" in str(e)
    flutes.exception.log = original_log


# LLM-generated content at query #17
#--------------------------

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
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_with_matching_arguments():
    captured_args = {}
    def custom_handler(e, arg1, arg2):
        captured_args['e'] = e
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("error")
    func_with_args(10, "hello")
    assert isinstance(captured_args['e'], RuntimeError)
    assert captured_args['arg1'] == 10
    assert captured_args['arg2'] == "hello"

def test_exception_wrapper_with_kwargs():
    captured = {}
    def custom_handler(e, my_arg, **kwargs):
        captured['e'] = e
        captured['my_arg'] = my_arg
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func_with_kwargs(a, b=5, **kwargs):
        raise TypeError("type error")
    func_with_kwargs(1, extra="value")
    assert isinstance(captured['e'], TypeError)
    assert captured['my_arg'] == 1
    assert captured['kwargs'] == {'b': 5, 'kwargs': {'extra': 'value'}}

def test_exception_wrapper_with_default_values_in_handler():
    captured = {}
    def custom_handler(e, required, optional=42):
        captured['e'] = e
        captured['required'] = required
        captured['optional'] = optional
    @exception_wrapper(custom_handler)
    def func_simple(x):
        raise ValueError("fail")
    func_simple(99)
    assert captured['required'] == 99
    assert captured['optional'] == 42

def test_exception_wrapper_with_generator():
    error_raised = False
    def custom_handler(e):
        nonlocal error_raised
        error_raised = True
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = gen_func()
    result = list(gen)
    assert result == [1]
    assert error_raised

def test_exception_wrapper_with_nested_wraps():
    def custom_handler(e, x):
        pass
    @exception_wrapper(custom_handler)
    @functools.lru_cache(maxsize=1)
    def func(x):
        raise ValueError("nested")
    func(5)

def test_exception_wrapper_raises_on_varargs_in_handler():
    def invalid_handler(e, *args):
        pass
    try:
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_raises_on_missing_handler_argument():
    def handler_missing(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler_missing)
        def func():
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_raises_on_default_arg_matching():
    def handler_with_default(e, arg_with_default=10):
        pass
    try:
        @exception_wrapper(handler_with_default)
        def func(arg_with_default):
            pass
        assert False
    except ValueError:
        pass

def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def normal_func():
        return "success"
    result = normal_func()
    assert result == "success"

def test_exception_wrapper_preserves_generator_yield():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg():
    def handler_without_exception_arg():
        pass

    try:
        @exception_wrapper(handler_without_exception_arg)
        def dummy():
            pass
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #19
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler_with_matching_args():
    caught_exception = None
    def handler(e, arg1, arg2):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("custom error")
    func("a", "b")
    assert caught_exception is not None
    assert isinstance(caught_exception, RuntimeError)

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def handler(e, my_arg=None, **kw):
        captured['e'] = e
        captured['my_arg'] = my_arg
        captured['kw'] = kw
    @exception_wrapper(handler)
    def func(one, two, *args, three=None, **kwargs):
        raise ValueError("error")
    func(1, 2, "extra", three=3, four=4)
    assert isinstance(captured['e'], ValueError)
    assert captured['my_arg'] is None
    assert captured['kw'] == {'one': 1, 'two': 2, 'args': ("extra",), 'three': 3, 'kwargs': {'four': 4}}

def test_exception_wrapper_generator_function():
    log_messages = []
    def mock_log(msg, level, **kwargs):
        log_messages.append((msg, level))
    import flutes.log
    original_log = flutes.log.log
    flutes.log.log = mock_log
    try:
        @exception_wrapper()
        def gen_func():
            yield 1
            raise ValueError("generator error")
            yield 2
        list(gen_func())
    finally:
        flutes.log.log = original_log
    assert any("generator error" in msg for msg, level in log_messages)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_handler_with_default_args():
    handler_called = False
    def handler(e, required_arg, optional_arg="default"):
        nonlocal handler_called
        handler_called = True
        assert required_arg == "req"
        assert optional_arg == "default"
    @exception_wrapper(handler)
    def func(required_arg):
        raise Exception("fail")
    func("req")
    assert handler_called

def test_exception_wrapper_raises_on_handler_varargs():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_raises_on_handler_no_args():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e)

def test_exception_wrapper_raises_on_mismatched_arg_without_default():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_raises_on_matched_arg_with_default():
    def handler(e, existing_arg="default"):
        pass
    try:
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_nested_wrapped_function():
    import functools
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @decorator
    def inner():
        raise ValueError("nested error")
    wrapped_func = exception_wrapper()(inner)
    wrapped_func()

def test_exception_wrapper_log_exception_failure():
    import flutes.log
    original_log = flutes.log.log
    def failing_log(msg, level, **kwargs):
        raise RuntimeError("log failed")
    flutes.log.log = failing_log
    try:
        @exception_wrapper()
        def func():
            raise ValueError("original error")
        func()
    except RuntimeError as e:
        assert "log failed" in str(e)
    finally:
        flutes.log.log = original_log


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorated = decorator(dummy_func)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped_arg():
    def handler_fn(e, arg1, arg2=5):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    try:
        func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #25
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    mock_log_calls = []
    def mock_log(msg, level="info", force_console=False, timestamp=True, include_proc_id=True):
        mock_log_calls.append((msg, level))
    import flutes.exception
    flutes.exception.log = mock_log
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    log_exception(e)
    assert len(mock_log_calls) == 1
    assert mock_log_calls[0][1] == "error"
    assert "<CalledProcessError>" in mock_log_calls[0][0]
    flutes.exception.log = log


# LLM-generated content at query #26
#--------------------------

def test_exception_wrapper_no_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()


def test_exception_wrapper_custom_handler():
    handler_called = False
    def handler(e):
        nonlocal handler_called
        handler_called = True
    @exception_wrapper(handler)
    def func():
        raise ValueError("test error")
    func()
    assert handler_called


def test_exception_wrapper_handler_with_matching_args():
    captured_args = {}
    def handler(e, arg1, arg2):
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("test error")
    func(1, 2)
    assert captured_args['arg1'] == 1
    assert captured_args['arg2'] == 2


def test_exception_wrapper_handler_with_default_args():
    captured_args = {}
    def handler(e, arg1, arg2, my_arg=None):
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
        captured_args['my_arg'] = my_arg
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("test error")
    func(1, 2)
    assert captured_args['arg1'] == 1
    assert captured_args['arg2'] == 2
    assert captured_args['my_arg'] is None


def test_exception_wrapper_handler_with_kwargs():
    captured_args = {}
    def handler(e, arg1, **kw):
        captured_args['arg1'] = arg1
        captured_args['kw'] = kw
    @exception_wrapper(handler)
    def func(arg1, arg2, **kwargs):
        raise ValueError("test error")
    func(1, 2, extra=3)
    assert captured_args['arg1'] == 1
    assert captured_args['kw'] == {'arg2': 2, 'kwargs': {'extra': 3}}


def test_exception_wrapper_handler_with_args_and_kwargs():
    captured_args = {}
    def handler(e, arg1, *args, **kw):
        captured_args['arg1'] = arg1
        captured_args['args'] = args
        captured_args['kw'] = kw
    @exception_wrapper(handler)
    def func(arg1, *args, **kwargs):
        raise ValueError("test error")
    func(1, 2, 3, extra=4)
    assert captured_args['arg1'] == 1
    assert captured_args['args'] == (2, 3)
    assert captured_args['kw'] == {'kwargs': {'extra': 4}}


def test_exception_wrapper_generator():
    @exception_wrapper()
    def func():
        yield 1
        raise ValueError("test error")
    gen = func()
    assert list(gen) == [1]


def test_exception_wrapper_generator_with_handler():
    handler_called = False
    def handler(e):
        nonlocal handler_called
        handler_called = True
    @exception_wrapper(handler)
    def func():
        yield 1
        raise ValueError("test error")
    gen = func()
    list(gen)
    assert handler_called


def test_exception_wrapper_return_value():
    @exception_wrapper()
    def func():
        return 42
    result = func()
    assert result == 42


def test_exception_wrapper_generator_return_value():
    @exception_wrapper()
    def func():
        yield from [1, 2, 3]
    gen = func()
    assert list(gen) == [1, 2, 3]


# LLM-generated content at query #27
#--------------------------

def test_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_is_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    skip_exceptions = [BdbQuit, KeyboardInterrupt]
    assert any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #28
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
    with patch('flutes.log.log') as mock_log:
        with patch('traceback.format_exc') as mock_format_exc:
            mock_format_exc.return_value = "traceback"
            log_exception(e)
            mock_log.assert_any_call("traceback", "error")
            mock_log.assert_any_call("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")
    e = ValueError("test")
    with patch('flutes.log.log') as mock_log:
        with patch('traceback.format_exc') as mock_format_exc:
            mock_format_exc.return_value = "traceback"
            log_exception(e)
            mock_log.assert_any_call("traceback", "error")
            mock_log.assert_any_call("<ValueError> test", "error")


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs_raises_error():
    def handler_with_varargs(e, *args):
        pass
    decorator = exception_wrapper(handler_with_varargs)
    def dummy_func():
        pass
    try:
        decorated = decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_with_custom_handler():
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

def test_exception_wrapper_passes_correct_arguments_to_handler():
    handler_args = {}
    def handler(e, one, two, three=None):
        handler_args['e'] = e
        handler_args['one'] = one
        handler_args['two'] = two
        handler_args['three'] = three
    @exception_wrapper(handler)
    def func(one, two, three=None):
        raise RuntimeError("error")
    func(1, 2, three=3)
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['one'] == 1
    assert handler_args['two'] == 2
    assert handler_args['three'] == 3

def test_exception_wrapper_with_var_kwargs():
    handler_kwargs = {}
    def handler(e, **kwargs):
        handler_kwargs.update(kwargs)
    @exception_wrapper(handler)
    def func(a, b, c=10):
        raise Exception()
    func(1, b=2)
    assert handler_kwargs == {'a': 1, 'b': 2, 'c': 10}

def test_exception_wrapper_with_matching_and_non_matching_args():
    handler_args = {}
    def handler(e, x, y, z=100):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['y'] = y
        handler_args['z'] = z
    @exception_wrapper(handler)
    def func(x, y):
        raise ValueError()
    func(x=5, y=6)
    assert handler_args['x'] == 5
    assert handler_args['y'] == 6
    assert handler_args['z'] == 100

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
    gen = gen_func()
    result = list(gen)
    assert result == [1]

def test_exception_wrapper_with_generator_and_custom_handler():
    caught = None
    def handler(e):
        nonlocal caught
        caught = e
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
    gen = gen_func()
    result = list(gen)
    assert result == [1]
    assert isinstance(caught, ValueError)
    assert str(caught) == "generator error"

def test_exception_wrapper_returns_non_generator():
    @exception_wrapper()
    def normal_func():
        return 42
    result = normal_func()
    assert result == 42

def test_exception_wrapper_raises_on_handler_with_varargs():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_raises_on_handler_without_exception_arg():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e)

def test_exception_wrapper_raises_on_missing_handler_arg():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_raises_on_handler_arg_with_default_matching_wrapped():
    def handler(e, arg_with_default=10):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg_with_default):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_wrapped_function():
    def inner():
        raise ValueError("inner")
    @exception_wrapper()
    @functools.wraps(inner)
    def wrapped():
        inner()
    wrapped()

def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = False
    def handler(e, a, b, *, kwonly=5):
        nonlocal handler_called
        handler_called = True
        assert a == 1
        assert b == 2
        assert kwonly == 5
    @exception_wrapper(handler)
    def func(a, b):
        raise Exception()
    func(1, 2)
    assert handler_called

def test_exception_wrapper_handler_captures_extra_kwargs():
    captured = {}
    def handler(e, a, **extra):
        captured.update(extra)
    @exception_wrapper(handler)
    def func(a, b, c=3):
        raise Exception()
    func(1, b=2)
    assert captured == {'b': 2, 'c': 3}

def test_exception_wrapper_with_positional_and_keyword_args():
    handler_args = {}
    def handler(e, pos1, pos2, kw1=None):
        handler_args['pos1'] = pos1
        handler_args['pos2'] = pos2
        handler_args['kw1'] = kw1
    @exception_wrapper(handler)
    def func(pos1, pos2, kw1=None):
        raise Exception()
    func(10, 20, kw1=30)
    assert handler_args['pos1'] == 10
    assert handler_args['pos2'] == 20
    assert handler_args['kw1'] == 30


# LLM-generated content at query #32
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    result = register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_handler_without_matching_args_can_have_defaults():
    from flutes.exception import exception_wrapper
    import inspect

    def handler_fn(e, non_matching_arg=None):
        pass

    @exception_wrapper(handler_fn)
    def foo():
        pass

    handler_argspec = inspect.getfullargspec(handler_fn)
    handler_arg_names = set(handler_argspec.args[1:] + handler_argspec.kwonlyargs)
    handler_args_with_defaults = set((handler_argspec.kwonlydefaults or {}).keys())
    if handler_argspec.defaults is not None:
        handler_args_with_defaults |= set(handler_argspec.args[-len(handler_argspec.defaults):])
    handler_arg_names -= handler_args_with_defaults
    inner_signature = inspect.signature(foo)
    for name in handler_args_with_defaults:
        if name in inner_signature.parameters:
            raise ValueError(f"Argument '{name}' matches wrapped method argument, thus cannot have default values")


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_handler_without_varargs():
    def handler_without_varargs(e, arg1):
        pass

    @exception_wrapper(handler_without_varargs)
    def func_with_arg(arg1):
        pass

    func_with_arg(1)


# LLM-generated content at query #36
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()

def test_exception_wrapper_custom_handler():
    def handler(e, x):
        assert isinstance(e, ValueError)
        assert x == 5
    @exception_wrapper(handler)
    def func(x):
        raise ValueError("test")
    func(5)

def test_exception_wrapper_handler_with_default():
    def handler(e, x, y=10):
        assert isinstance(e, ValueError)
        assert x == 5
        assert y == 10
    @exception_wrapper(handler)
    def func(x):
        raise ValueError("test")
    func(5)

def test_exception_wrapper_handler_with_kwargs():
    def handler(e, x, **kw):
        assert isinstance(e, ValueError)
        assert x == 5
        assert kw == {"y": 10}
    @exception_wrapper(handler)
    def func(x, **kwargs):
        raise ValueError("test")
    func(5, y=10)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func(x):
        return x * 2
    result = func(5)
    assert result == 10

def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
    gen = gen_func()
    items = list(gen)
    assert items == [1]

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    gen = gen_func()
    items = list(gen)
    assert items == [0, 1, 2]

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

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        exception_wrapper(handler)
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_default_matches_wrapped():
    def handler(e, x, y=10):
        pass
    try:
        @exception_wrapper(handler)
        def func(x, y):
            pass
        func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_with_kwonly():
    def handler(e, x, *, kwonly=20):
        assert isinstance(e, ValueError)
        assert x == 5
        assert kwonly == 20
    @exception_wrapper(handler)
    def func(x):
        raise ValueError("test")
    func(5)

def test_exception_wrapper_handler_with_args_and_kwargs():
    def handler(e, x, *args, **kwargs):
        pass
    try:
        exception_wrapper(handler)
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_log_exception_called():
    @exception_wrapper()
    def func():
        raise RuntimeError("logged")
    func()

def test_exception_wrapper_nested_wrapped():
    def inner_decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper()
    @inner_decorator
    def func():
        raise ValueError("nested")
    func()

def test_exception_wrapper_handler_with_mixed_args():
    def handler(e, a, b, c=30):
        assert isinstance(e, ValueError)
        assert a == 1
        assert b == 2
        assert c == 30
    @exception_wrapper(handler)
    def func(a, b):
        raise ValueError("test")
    func(1, 2)

def test_exception_wrapper_handler_with_only_exception_arg():
    def handler(e):
        assert isinstance(e, ValueError)
    @exception_wrapper(handler)
    def func():
        raise ValueError("test")
    func()


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg_raises_value_error():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #38
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


# LLM-generated content at query #39
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


# LLM-generated content at query #40
#--------------------------

def test_register_ipython_excepthook_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in skip_exceptions

def test_register_ipython_excepthook_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions

def test_register_ipython_excepthook_excepthook_skips_keyboard_interrupt_when_capture_keyboard_interrupt_false():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    mock_type = KeyboardInterrupt
    mock_value = KeyboardInterrupt()
    mock_traceback = None
    excepthook(mock_type, mock_value, mock_traceback)
    assert sys.__excepthook__ == sys.excepthook

def test_register_ipython_excepthook_excepthook_does_not_skip_keyboard_interrupt_when_capture_keyboard_interrupt_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    mock_type = KeyboardInterrupt
    mock_value = KeyboardInterrupt()
    mock_traceback = None
    excepthook(mock_type, mock_value, mock_traceback)
    assert ipython_hook == sys.excepthook


# LLM-generated content at query #41
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    class BdbQuit(BaseException):
        pass
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = True
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    excepthook_type = None
    excepthook_value = None
    excepthook_traceback = None
    def mock_excepthook(type, value, traceback):
        nonlocal excepthook_type, excepthook_value, excepthook_traceback
        excepthook_type = type
        excepthook_value = value
        excepthook_traceback = traceback
        if any(type is exc_type for exc_type in skip_exceptions):
            sys.__excepthook__(type, value, traceback)
        else:
            pass
    sys.excepthook = mock_excepthook
    test_exception = KeyboardInterrupt()
    sys.excepthook(type(test_exception), test_exception, None)
    assert excepthook_type is type(test_exception)
    assert excepthook_value is test_exception
    assert excepthook_traceback is None
    assert any(type(test_exception) is exc_type for exc_type in skip_exceptions) == False


# LLM-generated content at query #42
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import cast
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    excepthook = sys.excepthook
    mock_ipython_hook = Mock()
    with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
        excepthook(KeyboardInterrupt, KeyboardInterrupt(), cast(TracebackType, None))
    mock_ipython_hook.assert_called_once()


# LLM-generated content at query #43
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e, user_msg="Subprocess error")
    try:
        raise RuntimeError("runtime")
    except RuntimeError as e:
        log_exception(e, user_msg=None)
    try:
        raise TypeError("type")
    except TypeError as e:
        log_exception(e, force_console=True)
    try:
        raise KeyError("key")
    except KeyError as e:
        log_exception(e, timestamp=False)
    try:
        raise IndexError("index")
    except IndexError as e:
        log_exception(e, include_proc_id=False)


# LLM-generated content at query #2
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
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e)
    assert True

def test_log_exception_called_process_error_with_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    assert True

def test_log_exception_called_process_error_without_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=None)
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
        raise KeyError("key error")
    except KeyError as e:
        try:
            log_exception(e)
        except RuntimeError as log_e:
            assert str(log_e) == "log failed"
    log = original_log

def test_log_exception_with_kwargs():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise IndexError("index error")
    except IndexError as e:
        log_exception(e, force_console=True, timestamp=False)
    assert True


# LLM-generated content at query #3
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    captured_exception = None
    def custom_handler(e):
        nonlocal captured_exception
        captured_exception = e
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert captured_exception is not None
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured = {}
    def custom_handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise ValueError("error")
    func(1, 2)
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 1
    assert captured['arg2'] == 2

def test_exception_wrapper_custom_handler_with_default_args():
    captured = {}
    def custom_handler(e, arg1, arg2, extra=None):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['extra'] = extra
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise ValueError("error")
    func(1, 2)
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 1
    assert captured['arg2'] == 2
    assert captured['extra'] is None

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def custom_handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, **kwargs):
        raise ValueError("error")
    func(1, 2, extra=3)
    assert isinstance(captured['e'], ValueError)
    assert captured['arg1'] == 1
    assert captured['kwargs'] == {'arg2': 2, 'kwargs': {'extra': 3}}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func():
        return 42
    result = func()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def func():
        yield 1
        yield 2
    gen = func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_generator_with_exception():
    captured_exception = None
    def custom_handler(e):
        nonlocal captured_exception
        captured_exception = e
    @exception_wrapper(custom_handler)
    def func():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert captured_exception is not None
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "generator error"

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_invalid_handler_missing_arg():
    try:
        def custom_handler(e, missing_arg):
            pass
        @exception_wrapper(custom_handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_invalid_handler_default_arg_matches():
    try:
        def custom_handler(e, arg1, arg2=None):
            pass
        @exception_wrapper(custom_handler)
        def func(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #4
#--------------------------

def test_log_exception_called_process_error_with_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output="output")
    result = not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)
    assert result == False


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception_arg():
        pass
    try:
        @exception_wrapper(handler_without_exception_arg)
        def foo():
            pass
        foo()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #6
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
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e)
    except Exception as e:
        assert str(e) == "Log failure"
    finally:
        log = original_log

def test_log_exception_with_additional_kwargs():
    try:
        raise TypeError("Type mismatch")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)


# LLM-generated content at query #7
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

def test_register_ipython_excepthook_skip_exceptions():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    mock_traceback = None
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true_skip_exceptions():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    mock_traceback = None
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
    sys.excepthook = original_hook


# LLM-generated content at query #8
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    original_excepthook = sys.__excepthook__
    try:
        sys.excepthook = original_excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        skip_exceptions = sys.excepthook.__closure__[0].cell_contents
        assert KeyboardInterrupt not in skip_exceptions
    finally:
        sys.excepthook = original_excepthook

def test_skip_exceptions_contains_keyboard_interrupt_when_capture_keyboard_interrupt_is_false():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    original_excepthook = sys.__excepthook__
    try:
        sys.excepthook = original_excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        skip_exceptions = sys.excepthook.__closure__[0].cell_contents
        assert KeyboardInterrupt in skip_exceptions
    finally:
        sys.excepthook = original_excepthook

def test_excepthook_calls_sys_excepthook_for_skip_exceptions():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    original_excepthook = sys.__excepthook__
    mock_sys_excepthook = Mock()
    try:
        sys.__excepthook__ = mock_sys_excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        excepthook = sys.excepthook
        excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        mock_sys_excepthook.assert_called_once_with(KeyboardInterrupt, KeyboardInterrupt(), None)
    finally:
        sys.__excepthook__ = original_excepthook

def test_excepthook_calls_ipython_hook_for_non_skip_exceptions():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    original_excepthook = sys.__excepthook__
    mock_ipython_hook = Mock()
    try:
        sys.__excepthook__ = Mock()
        with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ipython_hook):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            excepthook = sys.excepthook
            excepthook(ValueError, ValueError(), None)
            mock_ipython_hook.assert_called_once_with(ValueError, ValueError(), None)
    finally:
        sys.__excepthook__ = original_excepthook

def test_skip_exceptions_always_contains_bdbquit():
    import sys
    from unittest.mock import Mock, patch
    from types import TracebackType
    from typing import Type
    from bdb import BdbQuit
    original_excepthook = sys.__excepthook__
    try:
        sys.excepthook = original_excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        skip_exceptions = sys.excepthook.__closure__[0].cell_contents
        assert BdbQuit in skip_exceptions
    finally:
        sys.excepthook = original_excepthook


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_custom_handler_with_matching_args():
    captured_exc = None
    captured_one = None
    captured_two = None
    def handler(e, one, two):
        nonlocal captured_exc, captured_one, captured_two
        captured_exc = e
        captured_one = one
        captured_two = two
    @exception_wrapper(handler)
    def func(one, two):
        raise RuntimeError("custom error")
    func(1, two=2)
    assert isinstance(captured_exc, RuntimeError)
    assert captured_one == 1
    assert captured_two == 2

def test_exception_wrapper_custom_handler_with_default_args():
    captured_exc = None
    captured_one = None
    captured_extra = None
    def handler(e, one, extra="default"):
        nonlocal captured_exc, captured_one, captured_extra
        captured_exc = e
        captured_one = one
        captured_extra = extra
    @exception_wrapper(handler)
    def func(one):
        raise KeyError("key error")
    func(one=10)
    assert isinstance(captured_exc, KeyError)
    assert captured_one == 10
    assert captured_extra == "default"

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_exc = None
    captured_kw = None
    def handler(e, **kw):
        nonlocal captured_exc, captured_kw
        captured_exc = e
        captured_kw = kw
    @exception_wrapper(handler)
    def func(a, b, c=3):
        raise TypeError("type error")
    func(1, 2)
    assert isinstance(captured_exc, TypeError)
    assert captured_kw == {"a": 1, "b": 2, "c": 3}

def test_exception_wrapper_custom_handler_mixed_args():
    captured_exc = None
    captured_a = None
    captured_kw = None
    def handler(e, a, extra="extra", **kw):
        nonlocal captured_exc, captured_a, captured_kw
        captured_exc = e
        captured_a = a
        captured_kw = kw
    @exception_wrapper(handler)
    def func(a, b):
        raise ValueError("mixed error")
    func(a=5, b=6)
    assert isinstance(captured_exc, ValueError)
    assert captured_a == 5
    assert captured_kw == {"b": 6}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_generator_with_exception():
    captured_exc = None
    def handler(e):
        nonlocal captured_exc
        captured_exc = e
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert isinstance(captured_exc, ValueError)

def test_exception_wrapper_wrapped_function():
    def dummy_decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    captured_exc = None
    def handler(e):
        nonlocal captured_exc
        captured_exc = e
    @exception_wrapper(handler)
    @dummy_decorator
    def func():
        raise RuntimeError("wrapped error")
    func()
    assert isinstance(captured_exc, RuntimeError)

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_handler_no_args_error():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e)

def test_exception_wrapper_handler_arg_mismatch_error():
    def handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_default_arg_matches_error():
    def handler(e, arg_with_default="default"):
        pass
    @exception_wrapper(handler)
    def func(arg_with_default):
        pass
    try:
        func(1)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #11
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
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), "error")
        mock_log.assert_any_call("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")
    e = ValueError("test")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), "error")
        mock_log.assert_any_call("<ValueError> test", "error")


# LLM-generated content at query #12
#--------------------------

def test_log_exception_with_subprocess_called_process_error_and_output_not_none():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    mock_e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    log_calls = []
    original_log = log
    log = lambda msg, level, **kwargs: log_calls.append((msg, level, kwargs))
    traceback.format_exc = lambda: "traceback"
    log_exception(mock_e)
    log = original_log
    assert len(log_calls) == 1
    assert log_calls[0][0] == "<CalledProcessError> Command '['ls']' returned non-zero exit status 1."
    assert log_calls[0][1] == "error"


# LLM-generated content at query #13
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=None)
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), "error")
        mock_log.assert_any_call("<CalledProcessError> Command 'test' returned non-zero exit status 1.", "error")
    e = ValueError("test")
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
        mock_log.assert_any_call(traceback.format_exc(), "error")
        mock_log.assert_any_call("<ValueError> test", "error")


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    log_exception(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_arg():
    def handler_without_exception_arg():
        pass

    try:
        exception_wrapper(handler_without_exception_arg)(lambda: None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")
    func()


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_raises_value_error_when_handler_fn_has_no_positional_argument():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    raised_error = None
    try:
        decorator(dummy_func)
    except ValueError as e:
        raised_error = e
    assert raised_error is not None
    assert "Exception handler must have a positional argument for the exception object" in str(raised_error)


# LLM-generated content at query #21
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
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_custom_handler_with_matching_args():
    captured_args = {}
    def handler(e, arg1, arg2):
        captured_args['e'] = e
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise ValueError("test error")
    func_raises(1, arg2=2)
    assert isinstance(captured_args['e'], ValueError)
    assert captured_args['arg1'] == 1
    assert captured_args['arg2'] == 2

def test_exception_wrapper_custom_handler_with_default_args():
    captured_args = {}
    def handler(e, arg1, arg2, my_default=None):
        captured_args['e'] = e
        captured_args['arg1'] = arg1
        captured_args['arg2'] = arg2
        captured_args['my_default'] = my_default
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise ValueError("test error")
    func_raises(1, arg2=2)
    assert isinstance(captured_args['e'], ValueError)
    assert captured_args['arg1'] == 1
    assert captured_args['arg2'] == 2
    assert captured_args['my_default'] is None

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_args = {}
    def handler(e, arg1, **kw):
        captured_args['e'] = e
        captured_args['arg1'] = arg1
        captured_args['kw'] = kw
    @exception_wrapper(handler)
    def func_raises(arg1, arg2, **kwargs):
        raise ValueError("test error")
    func_raises(1, arg2=2, extra=3)
    assert isinstance(captured_args['e'], ValueError)
    assert captured_args['arg1'] == 1
    assert captured_args['kw'] == {'arg2': 2, 'kwargs': {'extra': 3}}

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def func_gen():
        yield from range(3)
    gen = func_gen()
    assert list(gen) == [0, 1, 2]

def test_exception_wrapper_generator_exception():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_gen():
        yield 1
        raise ValueError("generator error")
        yield 2
    gen = func_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_handler_missing_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)

def test_exception_wrapper_handler_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_unmatched_arg():
    try:
        def handler(e, unmatched):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_matched_arg_with_default():
    try:
        def handler(e, arg1, default_arg=10):
            pass
        @exception_wrapper(handler)
        def func(arg1):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        pass
    inner.__wrapped__ = lambda: None
    def handler(e):
        pass
    decorator = exception_wrapper(handler)
    wrapped = decorator(inner)
    assert callable(wrapped)


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
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


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception():
        pass

    @exception_wrapper(handler_without_exception)
    def dummy():
        pass

    try:
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #24
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_custom_handler_with_matching_args():
    caught_exception = None
    def handler(e, arg1, arg2):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("custom error")
    func(1, 2)
    assert caught_exception is not None
    assert isinstance(caught_exception, RuntimeError)

def test_exception_wrapper_custom_handler_with_default_args():
    handler_called = False
    def handler(e, arg, extra=None):
        nonlocal handler_called
        handler_called = True
    @exception_wrapper(handler)
    def func(arg):
        raise Exception()
    func(5)
    assert handler_called

def test_exception_wrapper_custom_handler_with_kwargs():
    captured_kwargs = {}
    def handler(e, arg, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(handler)
    def func(arg, kw1=None, **extra):
        raise ValueError()
    func(10, kw1="value", extra1=1, extra2=2)
    assert captured_kwargs == {"kw1": "value", "extra": {"extra1": 1, "extra2": 2}}

def test_exception_wrapper_generator_function():
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
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func_normal():
        yield 1
        yield 2
    gen = gen_func_normal()
    assert list(gen) == [1, 2]

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_arg_mismatch():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func(existing_arg):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_default_arg_matches():
    try:
        def handler(e, arg, default_arg=5):
            pass
        @exception_wrapper(handler)
        def func(arg, default_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise ValueError("inner")
    wrapped = exception_wrapper()(inner)
    wrapped()

def test_exception_wrapper_double_wrapped():
    def handler(e):
        pass
    def inner():
        raise ValueError()
    wrapped_once = exception_wrapper(handler)(inner)
    wrapped_twice = exception_wrapper(handler)(wrapped_once)
    wrapped_twice()

def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = False
    def handler(e, arg, *, kwonly):
        nonlocal handler_called
        handler_called = True
    @exception_wrapper(handler)
    def func(arg, kwonly):
        raise Exception()
    func(1, kwonly=2)
    assert handler_called

def test_exception_wrapper_handler_with_args_and_kwargs():
    captured = {}
    def handler(e, pos, **kwargs):
        nonlocal captured
        captured = {"pos": pos, "kwargs": kwargs}
    @exception_wrapper(handler)
    def func(pos, key=None, **extra):
        raise ValueError()
    func(10, key="value", extra_key=3)
    assert captured["pos"] == 10
    assert captured["kwargs"] == {"key": "value", "extra": {"extra_key": 3}}


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #26
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_raises():
        raise ValueError("test error")
    func_raises()

def test_exception_wrapper_with_custom_handler():
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

def test_exception_wrapper_with_matching_args():
    captured = {}
    def handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func_raises(arg1, arg2):
        raise RuntimeError("error")
    func_raises(10, arg2="hello")
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == "hello"

def test_exception_wrapper_with_kwargs():
    captured = {}
    def handler(e, arg1, **kw):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kw'] = kw
    @exception_wrapper(handler)
    def func_raises(arg1, arg2=None, **kwargs):
        raise KeyError("key")
    func_raises(5, arg2=2, extra=3)
    assert isinstance(captured['e'], KeyError)
    assert captured['arg1'] == 5
    assert captured['kw'] == {'arg2': 2, 'extra': 3}

def test_exception_wrapper_with_default_args_in_handler():
    captured = {}
    def handler(e, arg1, my_default=42):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['my_default'] = my_default
    @exception_wrapper(handler)
    def func_raises(arg1):
        raise TypeError("type")
    func_raises(99)
    assert isinstance(captured['e'], TypeError)
    assert captured['arg1'] == 99
    assert captured['my_default'] == 42

def test_exception_wrapper_with_generator():
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

def test_exception_wrapper_with_nested_wrapped():
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    @decorator
    def func_raises():
        raise ValueError("nested error")
    func_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "nested error"

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_normal():
        return "success"
    result = func_normal()
    assert result == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

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

def test_exception_wrapper_invalid_handler_arg_mismatch():
    try:
        def handler(e, missing_arg):
            pass
        @exception_wrapper(handler)
        def func(arg1):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_invalid_handler_default_arg_matches():
    try:
        def handler(e, arg1, default_arg=10):
            pass
        @exception_wrapper(handler)
        def func(arg1, default_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #27
#--------------------------

def test_register_ipython_excepthook_skip_exceptions_initialization():
    result = register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_handler_fn_validation():
    def handler_fn(e, arg1, arg2, kwarg1=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, kwarg1=None, **kwargs):
        pass

    func(1, 2, kwarg1=3, extra=4)


# LLM-generated content at query #29
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_log_exception_with_user_msg():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, "User message")
    assert True

def test_log_exception_without_user_msg():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
    assert True

def test_log_exception_called_process_error_with_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    except subprocess.CalledProcessError as e:
        log_exception(e)
    assert True

def test_log_exception_called_process_error_without_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise subprocess.CalledProcessError(1, "cmd", output=None)
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


# LLM-generated content at query #2
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
    captured = {}
    def handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(1, arg2=2)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 1
    assert captured['arg2'] == 2

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg1, **kw):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kw'] = kw
    @exception_wrapper(handler)
    def func(arg1, arg2, arg3=3):
        raise KeyError("key error")
    func(10, arg2=20)
    assert isinstance(captured['e'], KeyError)
    assert captured['arg1'] == 10
    assert captured['kw'] == {'arg2': 20, 'arg3': 3}

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def handler(e, arg1, my_default=100):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['my_default'] = my_default
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise TypeError("type error")
    func(5, 6)
    assert isinstance(captured['e'], TypeError)
    assert captured['arg1'] == 5
    assert captured['my_default'] == 100

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_normal():
        return 42
    result = func_normal()
    assert result == 42

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    gen = gen_func()
    assert list(gen) == [1, 2]

def test_exception_wrapper_generator_exception():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "gen error"

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e) or "*args" in str(e)

def test_exception_wrapper_handler_no_exception_arg():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e) or "exception object" in str(e)

def test_exception_wrapper_handler_arg_mismatch():
    def handler(e, non_existent_arg):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg1):
            pass
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches():
    def handler(e, arg1, arg2=10):
        pass
    try:
        @exception_wrapper(handler)
        def func(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise IndexError("inner error")
    wrapped = exception_wrapper()(inner)
    wrapped()

def test_exception_wrapper_already_wrapped():
    def handler(e):
        pass
    @exception_wrapper(handler)
    def inner():
        raise Exception("test")
    @exception_wrapper(handler)
    def outer():
        return inner()
    outer()


# LLM-generated content at query #3
#--------------------------

def test_log_exception_with_user_msg():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e, user_msg="Subprocess error")
    try:
        raise RuntimeError("runtime")
    except RuntimeError as e:
        log_exception(e, user_msg=None)
    try:
        raise KeyError("key")
    except KeyError as e:
        log_exception(e, force_console=True)
    try:
        raise TypeError("type")
    except TypeError as e:
        log_exception(e, timestamp=False)
    try:
        raise IndexError("index")
    except IndexError as e:
        log_exception(e, include_proc_id=False)


# LLM-generated content at query #4
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    mock_log_calls = []
    def mock_log(msg, level="info", **kwargs):
        mock_log_calls.append((msg, level))
    original_log = log
    log = mock_log
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    log_exception(e)
    log = original_log
    assert len(mock_log_calls) == 1
    assert mock_log_calls[0][1] == "error"
    assert "<CalledProcessError>" in mock_log_calls[0][0]


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_exception_arg():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #6
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises(x):
        raise ValueError(f"Error with {x}")
    func_that_raises(5)

def test_exception_wrapper_with_custom_handler():
    captured_exception = None
    captured_args = {}
    def custom_handler(e, x, y=10):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {'x': x, 'y': y}
    @exception_wrapper(custom_handler)
    def func(x, y=5):
        raise RuntimeError("Oops")
    func(3)
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args == {'x': 3, 'y': 5}

def test_exception_wrapper_with_kwargs():
    captured_kwargs = {}
    def custom_handler(e, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    @exception_wrapper(custom_handler)
    def func(a, b=2, **kwargs):
        raise KeyError("Key missing")
    func(1, c=3)
    assert captured_kwargs == {'a': 1, 'b': 2, 'kwargs': {'c': 3}}

def test_exception_wrapper_with_matching_args():
    captured = {}
    def custom_handler(e, x, z):
        nonlocal captured
        captured = {'x': x, 'z': z}
    @exception_wrapper(custom_handler)
    def func(x, y, z=10):
        raise TypeError("Type mismatch")
    func(7, 8)
    assert captured == {'x': 7, 'z': 10}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Bad value")
            yield i
    gen = gen_func(5)
    result = list(gen)
    assert result == [0, 1]

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def add(a, b):
        return a + b
    assert add(2, 3) == 5

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def bad_handler():
            pass
        @exception_wrapper(bad_handler)
        def dummy():
            pass
        dummy()
        assert False
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def bad_handler(e, *args):
            pass
        @exception_wrapper(bad_handler)
        def dummy():
            pass
        dummy()
        assert False
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_unmatched_arg():
    try:
        def handler(e, unmatched):
            pass
        @exception_wrapper(handler)
        def dummy(x):
            pass
        dummy(1)
        assert False
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_matched_arg_with_default():
    try:
        def handler(e, x, y=5):
            pass
        @exception_wrapper(handler)
        def dummy(x):
            pass
        dummy(1)
        assert False
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise AssertionError("Inner error")
    wrapped = exception_wrapper()(inner)
    wrapped()

def test_exception_wrapper_with_log_exception():
    @exception_wrapper()
    def failing():
        raise FileNotFoundError("File not found")
    failing()


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    from unittest.mock import patch, MagicMock
    e = subprocess.CalledProcessError(returncode=1, cmd='ls', output='some output')
    with patch('flutes.log.log') as mock_log:
        log_exception(e)
        mock_log.assert_called_once_with('<CalledProcessError> Command \'ls\' returned non-zero exit status 1.', 'error')
    e_no_output = subprocess.CalledProcessError(returncode=1, cmd='ls')
    with patch('flutes.log.log') as mock_log:
        log_exception(e_no_output)
        assert mock_log.call_count == 2
        first_call = mock_log.call_args_list[0]
        second_call = mock_log.call_args_list[1]
        assert first_call[0][1] == 'error'
        assert second_call[0][1] == 'error'
    e_other = ValueError('test')
    with patch('flutes.log.log') as mock_log:
        log_exception(e_other)
        assert mock_log.call_count == 2
        first_call = mock_log.call_args_list[0]
        second_call = mock_log.call_args_list[1]
        assert first_call[0][1] == 'error'
        assert second_call[0][1] == 'error'


# LLM-generated content at query #11
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    log_calls = []
    original_log = log
    log = lambda msg, level, **kwargs: log_calls.append((msg, level, kwargs))
    traceback.format_exc = lambda: "traceback"
    try:
        log_exception(e)
    finally:
        log = original_log
    assert len(log_calls) == 1
    assert log_calls[0][0] == f"<CalledProcessError> Command 'test' returned non-zero exit status 1."
    assert log_calls[0][1] == "error"


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(returncode=1, cmd="test", output=b"output")
    log_exception(e)


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_handler_without_positional_arg_raises_value_error():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #16
#--------------------------

def test_register_ipython_excepthook_default():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook != original_hook
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    hook = sys.excepthook
    mock_traceback = unittest.mock.Mock()
    with unittest.mock.patch('sys.__excepthook__') as mock_original:
        hook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
        mock_original.assert_called_once_with(KeyboardInterrupt, unittest.mock.ANY, mock_traceback)
    sys.excepthook = original_hook

def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    original_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    hook = sys.excepthook
    mock_traceback = unittest.mock.Mock()
    with unittest.mock.patch('sys.__excepthook__') as mock_original:
        hook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
        mock_original.assert_not_called()
    sys.excepthook = original_hook

def test_register_ipython_excepthook_skips_bdbquit():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    hook = sys.excepthook
    mock_traceback = unittest.mock.Mock()
    with unittest.mock.patch('sys.__excepthook__') as mock_original:
        hook(BdbQuit, BdbQuit(), mock_traceback)
        mock_original.assert_called_once_with(BdbQuit, unittest.mock.ANY, mock_traceback)
    sys.excepthook = original_hook

def test_register_ipython_excepthook_calls_ipython_for_other_exceptions():
    original_hook = sys.excepthook
    register_ipython_excepthook()
    hook = sys.excepthook
    mock_traceback = unittest.mock.Mock()
    with unittest.mock.patch('sys.__excepthook__') as mock_original:
        with unittest.mock.patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB:
            mock_instance = unittest.mock.Mock()
            MockFormattedTB.return_value = mock_instance
            register_ipython_excepthook()
            hook = sys.excepthook
            hook(ValueError, ValueError("test"), mock_traceback)
            mock_instance.assert_called_once_with(ValueError, unittest.mock.ANY, mock_traceback)
            mock_original.assert_not_called()
    sys.excepthook = original_hook


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_arg():
    def handler_without_exception_arg():
        pass
    try:
        exception_wrapper(handler_without_exception_arg)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_handler_without_exception_argument():
    def handler_without_exception():
        pass

    @exception_wrapper(handler_without_exception)
    def dummy():
        pass

    try:
        dummy()
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #19
#--------------------------

def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_handler():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_with_matching_args():
    args_captured = {}
    def handler(e, arg1, arg2):
        args_captured['arg1'] = arg1
        args_captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, arg2=20)
    assert args_captured['arg1'] == 10
    assert args_captured['arg2'] == 20

def test_exception_wrapper_with_default_args():
    args_captured = {}
    def handler(e, arg1, arg2, extra="default"):
        args_captured['arg1'] = arg1
        args_captured['arg2'] = arg2
        args_captured['extra'] = extra
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, arg2=15)
    assert args_captured['arg1'] == 5
    assert args_captured['arg2'] == 15
    assert args_captured['extra'] == "default"

def test_exception_wrapper_with_var_kwargs():
    args_captured = {}
    def handler(e, arg1, **kwargs):
        args_captured['arg1'] = arg1
        args_captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func(arg1, arg2, **kwargs):
        raise RuntimeError("error")
    func(1, arg2=2, extra=3)
    assert args_captured['arg1'] == 1
    assert args_captured['kwargs'] == {'arg2': 2, 'extra': 3}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("generator error")
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_normal_return():
    @exception_wrapper()
    def normal_func():
        return 42
    result = normal_func()
    assert result == 42

def test_exception_wrapper_handler_without_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_with_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_arg_mismatch():
    try:
        def handler(e, non_existent_arg):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_default_arg_matches():
    try:
        def handler(e, arg, default_arg="value"):
            pass
        @exception_wrapper(handler)
        def func(arg, default_arg):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_wrapped_function():
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @decorator
    @exception_wrapper()
    def func():
        raise ValueError("test")
    func()


# LLM-generated content at query #20
#--------------------------

def test_log_exception_with_called_process_error_and_output():
    import subprocess
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    mock_log_calls = []
    original_log = log
    def mock_log(msg, level="info", force_console=False, timestamp=True, include_proc_id=True):
        mock_log_calls.append((msg, level))
    log = mock_log
    e = subprocess.CalledProcessError(returncode=1, cmd=["ls"], output=b"some output")
    log_exception(e)
    log = original_log
    assert len(mock_log_calls) == 1
    assert mock_log_calls[0][1] == "error"
    assert "<CalledProcessError>" in mock_log_calls[0][0]


# LLM-generated content at query #21
#--------------------------

def test_skip_exceptions_does_not_contain_keyboard_interrupt_when_capture_keyboard_interrupt_is_true():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in skip_exceptions


# LLM-generated content at query #22
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
    try:
        raise subprocess.CalledProcessError(1, "cmd")
    except subprocess.CalledProcessError as e:
        log_exception(e, user_msg="Subprocess error")
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e, user_msg=None)
    try:
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)
    try:
        raise KeyError("key error")
    except KeyError as e:
        log_exception(e, include_proc_id=False)
    try:
        raise IndexError("index error")
    except IndexError as e:
        log_exception(e, level="warning")
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError as e:
        log_exception(e, user_msg="Math error", force_console=False, timestamp=True, include_proc_id=True)


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
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

def test_exception_wrapper_handler_with_matching_args():
    captured = {}
    def handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, arg2="hello")
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == "hello"

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def handler(e, arg1, arg2, extra="default"):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['extra'] = extra
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, arg2="world")
    assert captured['arg1'] == 5
    assert captured['arg2'] == "world"
    assert captured['extra'] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func(arg1, arg2, arg3=30):
        raise RuntimeError("error")
    func(1, arg2=2, arg3=3, extra=4)
    assert captured['arg1'] == 1
    assert captured['kwargs'] == {'arg2': 2, 'arg3': 3, 'extra': 4}

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

def test_exception_wrapper_generator_exception():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
        yield 2
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "gen error"

def test_exception_wrapper_handler_varargs_error():
    def handler(e, *args):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e) or "*args" in str(e)

def test_exception_wrapper_handler_no_exception_arg_error():
    def handler():
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_unmatched_arg_error():
    def handler(e, unmatched):
        pass
    try:
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_handler_matched_arg_with_default_error():
    def handler(e, arg1, arg2="default"):
        pass
    @exception_wrapper(handler)
    def func(arg1, arg2):
        pass
    func(1, 2)

def test_exception_wrapper_already_wrapped():
    def handler(e):
        pass
    @exception_wrapper(handler)
    @exception_wrapper(handler)
    def func():
        raise ValueError("test")
    func()


# LLM-generated content at query #25
#--------------------------

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
    def custom_handler(e, one, two):
        handler_args['e'] = e
        handler_args['one'] = one
        handler_args['two'] = two
    @exception_wrapper(custom_handler)
    def func(one, two):
        raise RuntimeError("error")
    func(1, two=2)
    assert isinstance(handler_args['e'], RuntimeError)
    assert handler_args['one'] == 1
    assert handler_args['two'] == 2

def test_exception_wrapper_with_kwargs():
    handler_kwargs = {}
    def custom_handler(e, **kwargs):
        handler_kwargs['e'] = e
        handler_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def func(a, b=10):
        raise Exception("error")
    func(5)
    assert isinstance(handler_kwargs['e'], Exception)
    assert handler_kwargs['a'] == 5
    assert handler_kwargs['b'] == 10

def test_exception_wrapper_with_args_and_kwargs():
    captured = {}
    def custom_handler(e, x, **kw):
        captured['e'] = e
        captured['x'] = x
        captured['kw'] = kw
    @exception_wrapper(custom_handler)
    def func(x, *args, y=20, **kwargs):
        raise ValueError("error")
    func(1, 2, 3, y=30, z=40)
    assert isinstance(captured['e'], ValueError)
    assert captured['x'] == 1
    assert captured['kw'] == {'args': (2, 3), 'y': 30, 'kwargs': {'z': 40}}

def test_exception_wrapper_with_generator():
    error_raised = False
    def custom_handler(e):
        nonlocal error_raised
        error_raised = True
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
        yield 2
    g = gen_func()
    assert list(g) == [1]
    assert error_raised

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    result = normal_func(5)
    assert result == 10

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    g = gen_func()
    assert list(g) == [0, 1, 2]

def test_exception_wrapper_handler_with_default_values():
    handler_called = False
    def custom_handler(e, required, optional=100):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, TypeError)
        assert required == "req"
        assert optional == 100
    @exception_wrapper(custom_handler)
    def func(required):
        raise TypeError("error")
    func("req")
    assert handler_called

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_unmatched_arg():
    try:
        def custom_handler(e, unmatched):
            pass
        @exception_wrapper(custom_handler)
        def func():
            pass
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_matched_arg_with_default():
    try:
        def custom_handler(e, matched=10):
            pass
        @exception_wrapper(custom_handler)
        def func(matched):
            pass
    except ValueError:
        pass

def test_exception_wrapper_wrapped_function():
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper()
    @decorator
    def func():
        raise ValueError("error")
    func()

def test_exception_wrapper_handler_with_kwonlyargs():
    handler_args = {}
    def custom_handler(e, a, *, kwonly):
        handler_args['e'] = e
        handler_args['a'] = a
        handler_args['kwonly'] = kwonly
    @exception_wrapper(custom_handler)
    def func(a, kwonly=None):
        raise Exception("error")
    func(42, kwonly="kw")
    assert isinstance(handler_args['e'], Exception)
    assert handler_args['a'] == 42
    assert handler_args['kwonly'] == "kw"


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg():
    def handler_without_arg():
        pass
    decorator = exception_wrapper(handler_without_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #27
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def raises_exception():
        raise ValueError("test error")
    raises_exception()

def test_exception_wrapper_with_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def raises_exception():
        raise ValueError("test error")
    raises_exception()
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_with_matching_arguments():
    handler_args = []
    def custom_handler(e, arg1, arg2):
        handler_args.extend([arg1, arg2])
    @exception_wrapper(custom_handler)
    def raises_exception(arg1, arg2):
        raise ValueError("test error")
    raises_exception(1, 2)
    assert handler_args == [1, 2]

def test_exception_wrapper_with_kwargs():
    handler_kwargs = {}
    def custom_handler(e, **kwargs):
        handler_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def raises_exception(arg1, arg2):
        raise ValueError("test error")
    raises_exception(1, arg2=2)
    assert handler_kwargs == {"arg1": 1, "arg2": 2}

def test_exception_wrapper_with_default_values():
    handler_args = []
    def custom_handler(e, arg1, arg2, optional="default"):
        handler_args.extend([arg1, arg2, optional])
    @exception_wrapper(custom_handler)
    def raises_exception(arg1, arg2):
        raise ValueError("test error")
    raises_exception(1, 2)
    assert handler_args == [1, 2, "default"]

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_raises():
        yield 1
        raise ValueError("generator error")
    gen = generator_raises()
    result = list(gen)
    assert result == [1]

def test_exception_wrapper_with_nested_wrapping():
    @exception_wrapper()
    @exception_wrapper()
    def raises_exception():
        raise ValueError("test error")
    raises_exception()

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def no_exception():
        return 42
    result = no_exception()
    assert result == 42

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def generator_no_exception():
        yield from range(3)
    gen = generator_no_exception()
    result = list(gen)
    assert result == [0, 1, 2]

def test_exception_wrapper_with_mismatched_argument():
    def custom_handler(e, non_existent_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_value_matching_argument():
    def custom_handler(e, arg1, arg2="default"):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(arg1, arg2):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_positional_argument():
    def custom_handler():
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #28
#--------------------------

def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(handler)
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "test error"

def test_exception_wrapper_handler_with_matching_args():
    captured = {}
    def handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(10, arg2=20)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == 20

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def handler(e, arg1, arg2, extra="default"):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['extra'] = extra
    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, arg2=15)
    assert captured['arg1'] == 5
    assert captured['arg2'] == 15
    assert captured['extra'] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(handler)
    def func(arg1, arg2, **kwargs):
        raise RuntimeError("error")
    func(1, arg2=2, extra=3)
    assert captured['arg1'] == 1
    assert captured['kwargs'] == {'arg2': 2, 'extra': 3}

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

def test_exception_wrapper_generator_exception():
    caught_exception = None
    def handler(e):
        nonlocal caught_exception
        caught_exception = e
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
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_handler_missing_exception_arg():
    try:
        def handler():
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_varargs():
    try:
        def handler(e, *args):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_unmatched_arg():
    try:
        def handler(e, unmatched):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_matched_arg_with_default():
    try:
        def handler(e, arg1="default"):
            pass
        @exception_wrapper(handler)
        def func(arg1):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        pass
    inner.__wrapped__ = lambda: None
    @exception_wrapper()
    def outer():
        pass
    outer.__wrapped__ = inner
    result = exception_wrapper._unwrap(outer)
    assert result is inner.__wrapped__


# LLM-generated content at query #29
#--------------------------

def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    captured_exception = None
    captured_args = {}
    def custom_handler(e, arg1, arg2, extra=None):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {"arg1": arg1, "arg2": arg2, "extra": extra}
    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("custom error")
    func_with_args("value1", arg2="value2")
    assert isinstance(captured_exception, RuntimeError)
    assert captured_args["arg1"] == "value1"
    assert captured_args["arg2"] == "value2"
    assert captured_args["extra"] is None

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

def test_exception_wrapper_with_nested_decorator():
    def custom_handler(e, x):
        pass
    @exception_wrapper(custom_handler)
    @functools.lru_cache(maxsize=1)
    def nested_func(x):
        raise AssertionError("nested error")
    nested_func(5)

def test_exception_wrapper_invalid_handler_no_args():
    try:
        @exception_wrapper(lambda: None)
        def dummy():
            pass
        dummy()
        assert False
    except ValueError:
        pass

def test_exception_wrapper_invalid_handler_varargs():
    try:
        @exception_wrapper(lambda e, *args: None)
        def dummy():
            pass
        dummy()
        assert False
    except ValueError:
        pass

def test_exception_wrapper_missing_handler_arg():
    try:
        @exception_wrapper(lambda e, missing_arg: None)
        def dummy():
            pass
        dummy()
        assert False
    except ValueError:
        pass

def test_exception_wrapper_handler_arg_with_default_matches():
    try:
        @exception_wrapper(lambda e, arg_with_default=5: None)
        def dummy(arg_with_default):
            pass
        dummy()
        assert False
    except ValueError:
        pass

def test_exception_wrapper_handler_with_matching_and_extra_args():
    captured = {}
    def custom_handler(e, matched, extra="default"):
        nonlocal captured
        captured = {"matched": matched, "extra": extra}
    @exception_wrapper(custom_handler)
    def func(matched):
        raise Exception("test")
    func(matched=42)
    assert captured["matched"] == 42
    assert captured["extra"] == "default"

def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def normal_func():
        return "success"
    result = normal_func()
    assert result == "success"

def test_exception_wrapper_preserves_generator_yield():
    @exception_wrapper()
    def safe_generator():
        yield from range(3)
    gen = safe_generator()
    assert list(gen) == [0, 1, 2]


# LLM-generated content at query #30
#--------------------------

def test_exception_wrapper_logs_exception():
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
        raise ValueError("custom error")
    func_that_raises()
    assert caught_exception is not None
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "custom error"

def test_exception_wrapper_handler_with_matching_args():
    captured = {}
    def custom_handler(e, arg1, arg2):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise RuntimeError("error with args")
    func(10, arg2=20)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 10
    assert captured['arg2'] == 20

def test_exception_wrapper_handler_with_default_args():
    captured = {}
    def custom_handler(e, arg1, arg2, optional="default"):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['arg2'] = arg2
        captured['optional'] = optional
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise Exception("test")
    func(1, 2)
    assert captured['arg1'] == 1
    assert captured['arg2'] == 2
    assert captured['optional'] == "default"

def test_exception_wrapper_handler_with_kwargs():
    captured = {}
    def custom_handler(e, arg1, **kwargs):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kwargs'] = kwargs
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, arg3=30):
        raise Exception("test")
    func(100, arg2=200)
    assert captured['arg1'] == 100
    assert captured['kwargs'] == {'arg2': 200, 'arg3': 30}

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

def test_exception_wrapper_generator_exception():
    captured_exception = None
    def custom_handler(e):
        nonlocal captured_exception
        captured_exception = e
    @exception_wrapper(custom_handler)
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
    assert captured_exception is not None
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "generator error"

def test_exception_wrapper_invalid_handler_no_args():
    try:
        def invalid_handler():
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_invalid_handler_varargs():
    try:
        def invalid_handler(e, *args):
            pass
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_arg_mismatch():
    try:
        def handler(e, non_existent_arg):
            pass
        @exception_wrapper(handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_default_arg_matches():
    try:
        def handler(e, arg_with_default=10):
            pass
        @exception_wrapper(handler)
        def func(arg_with_default):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function():
    def inner():
        raise ValueError("inner")
    wrapped = exception_wrapper()(inner)
    wrapped()

def test_exception_wrapper_already_wrapped():
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped
    @exception_wrapper()
    @decorator
    def func():
        raise ValueError("wrapped")
    func()

def test_exception_wrapper_log_exception_calledprocesserror():
    import subprocess
    @exception_wrapper()
    def func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"output")
    func()


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_handler_fn_without_positional_arg_raises_value_error():
    def handler_without_exception_arg():
        pass
    decorator = exception_wrapper(handler_without_exception_arg)
    def dummy_func():
        pass
    try:
        decorator(dummy_func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #32
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
    def custom_handler(e, arg1, my_default=None):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['my_default'] = my_default
    @exception_wrapper(custom_handler)
    def func(arg1, arg2):
        raise RuntimeError("error")
    func(5, arg2=15)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 5
    assert captured['my_default'] is None

def test_exception_wrapper_custom_handler_with_kwargs():
    captured = {}
    def custom_handler(e, arg1, **kw):
        captured['e'] = e
        captured['arg1'] = arg1
        captured['kw'] = kw
    @exception_wrapper(custom_handler)
    def func(arg1, arg2, **kwargs):
        raise RuntimeError("error")
    func(1, 2, extra=3)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['arg1'] == 1
    assert captured['kw'] == {'arg2': 2, 'kwargs': {'extra': 3}}

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

def test_exception_wrapper_generator_with_exception():
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
    try:
        next(gen)
    except StopIteration:
        pass
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "generator error"

def test_exception_wrapper_handler_missing_exception_arg():
    try:
        def handler_without_exception():
            pass
        @exception_wrapper(handler_without_exception)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_handler_varargs():
    try:
        def handler_with_varargs(e, *args):
            pass
        @exception_wrapper(handler_with_varargs)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_arg_mismatch():
    try:
        def custom_handler(e, non_existent_arg):
            pass
        @exception_wrapper(custom_handler)
        def func():
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_handler_default_arg_matches_wrapped():
    try:
        def custom_handler(e, arg1, arg2=None):
            pass
        @exception_wrapper(custom_handler)
        def func(arg1, arg2):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_wrapped_function_returns_generator():
    @exception_wrapper()
    def gen_func():
        yield from range(3)
    result = gen_func()
    assert list(result) == [0, 1, 2]

def test_exception_wrapper_log_exception_integration():
    @exception_wrapper()
    def func():
        raise ValueError("integration test")
    func()

def test_exception_wrapper_with_args_kwargs():
    captured = {}
    def custom_handler(e, a, b, c, d, **kw):
        captured.update(e=e, a=a, b=b, c=c, d=d, kw=kw)
    @exception_wrapper(custom_handler)
    def func(a, b, *args, c=None, **kwargs):
        raise RuntimeError("test")
    func(1, 2, 3, 4, c=5, extra=6)
    assert isinstance(captured['e'], RuntimeError)
    assert captured['a'] == 1
    assert captured['b'] == 2
    assert captured['c'] == 5
    assert captured['d'] == (3, 4)
    assert captured['kw'] == {'args': (3, 4), 'kwargs': {'extra': 6}}

def test_exception_wrapper_handler_with_only_exception_arg():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def func(x, y):
        raise ValueError("simple")
    func(100, 200)
    assert isinstance(caught_exception, ValueError)
    assert str(caught_exception) == "simple"


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_handler_fn_with_varargs_raises_value_error():
    def handler_fn(e, *args):
        pass
    decorator = exception_wrapper(handler_fn)
    def func():
        pass
    try:
        decorator(func)
        assert False
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_handler_arg_with_default_matches_wrapped_arg():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        pass

    try:
        func(1, 2)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #35
#--------------------------

def test_exception_wrapper_logs_exception():
    @exception_wrapper()
    def failing_function():
        raise ValueError("Test error")
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
        raise ValueError("Generator error")
    gen = generator_function()
    assert list(gen) == [1]

def test_exception_wrapper_custom_handler():
    caught_exception = None
    def custom_handler(e):
        nonlocal caught_exception
        caught_exception = e
    @exception_wrapper(custom_handler)
    def failing_function():
        raise RuntimeError("Custom handler test")
    failing_function()
    assert isinstance(caught_exception, RuntimeError)
    assert str(caught_exception) == "Custom handler test"

def test_exception_wrapper_handler_with_matching_args():
    handler_args = {}
    def custom_handler(e, arg1, arg2):
        handler_args['e'] = e
        handler_args['arg1'] = arg1
        handler_args['arg2'] = arg2
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise ValueError("Args test")
    failing_function(10, arg2=20)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['arg1'] == 10
    assert handler_args['arg2'] == 20

def test_exception_wrapper_handler_with_default_args():
    handler_called = False
    def custom_handler(e, arg1, arg2, optional_arg="default"):
        nonlocal handler_called
        handler_called = True
        assert arg1 == 5
        assert arg2 == 6
        assert optional_arg == "default"
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2):
        raise Exception()
    failing_function(5, 6)
    assert handler_called

def test_exception_wrapper_handler_with_kwargs():
    captured_kwargs = {}
    def custom_handler(e, arg1, **kwargs):
        captured_kwargs['e'] = e
        captured_kwargs['arg1'] = arg1
        captured_kwargs.update(kwargs)
    @exception_wrapper(custom_handler)
    def failing_function(arg1, arg2, **kwargs):
        raise ValueError("Kwargs test")
    failing_function(1, 2, extra=3)
    assert isinstance(captured_kwargs['e'], ValueError)
    assert captured_kwargs['arg1'] == 1
    assert captured_kwargs['arg2'] == 2
    assert captured_kwargs['extra'] == 3

def test_exception_wrapper_handler_missing_arg_raises_error():
    def custom_handler(e, missing_arg):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_handler_arg_with_default_matches_wrapped_raises_error():
    def custom_handler(e, arg1, arg2="default"):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function(arg1, arg2):
            pass
        assert False
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_handler_varargs_raises_error():
    def custom_handler(e, *args):
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_handler_no_args_raises_error():
    def custom_handler():
        pass
    try:
        @exception_wrapper(custom_handler)
        def some_function():
            pass
        assert False
    except ValueError as e:
        assert "must have a positional argument" in str(e)


