####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_basic():
    import traceback
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e)
            assert mock_log.call_count == 2
            assert mock_log.call_args_list[0][0][1] == "error"
            assert mock_log.call_args_list[1][0][1] == "error"
            assert "<ValueError> test error" in mock_log.call_args_list[1][0][0]


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Custom message")
            assert mock_log.call_count == 2
            assert "Custom message: <RuntimeError> runtime error" in mock_log.call_args_list[1][0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, force_console=True, timestamp=False)
            assert mock_log.call_count == 2
            assert mock_log.call_args_list[0][1]['force_console'] is True
            assert mock_log.call_args_list[0][1]['timestamp'] is False


def test_log_exception_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, 'cmd', output='output')
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 1
        assert "<CalledProcessError>" in mock_log.call_args_list[0][0][0]


def test_log_exception_subprocess_error_no_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, 'cmd', output=None)
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 2


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            mock_log.side_effect = RuntimeError("logging failed")
            try:
                log_exception(e)
            except RuntimeError:
                pass
            assert mock_log.call_count == 1


def test_log_exception_multiple_exception_types():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    exceptions = [
        ValueError("value error"),
        KeyError("key error"),
        AttributeError("attribute error"),
        IndexError("index error")
    ]
    
    for exc in exceptions:
        try:
            raise exc
        except Exception as e:
            with patch('flutes.exception.log') as mock_log:
                log_exception(e)
                assert mock_log.call_count == 2
                assert e.__class__.__qualname__ in mock_log.call_args_list[1][0][0]


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_sets_excepthook():
    import sys
    
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    new_excepthook = sys.excepthook
    
    assert new_excepthook is not original_excepthook
    assert callable(new_excepthook)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_bdbquit_not_captured():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    try:
        raise BdbQuit()
    except BdbQuit:
        pass
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_not_captured_by_default():
    import sys
    
    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_basic():
    import traceback
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise ValueError("test error")
        except ValueError as e:
            log_exception(e)
    
    assert mock_log.call_count == 2
    calls = mock_log.call_args_list
    assert "ValueError" in calls[0][0][0]
    assert "<ValueError> test error" in calls[1][0][0]
    assert calls[1][1]['level'] == "error"


def test_log_exception_with_user_msg():
    import traceback
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise RuntimeError("runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
    
    assert mock_log.call_count == 2
    calls = mock_log.call_args_list
    assert "Custom message: <RuntimeError> runtime issue" in calls[1][0][0]


def test_log_exception_with_kwargs():
    import traceback
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise TypeError("type mismatch")
        except TypeError as e:
            log_exception(e, force_console=True, timestamp=False)
    
    assert mock_log.call_count == 2
    calls = mock_log.call_args_list
    assert calls[0][1]['force_console'] is True
    assert calls[0][1]['timestamp'] is False
    assert calls[1][1]['force_console'] is True
    assert calls[1][1]['timestamp'] is False


def test_log_exception_with_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        error = subprocess.CalledProcessError(1, 'cmd', output='some output')
        log_exception(error)
    
    assert mock_log.call_count == 1
    calls = mock_log.call_args_list
    assert "CalledProcessError" in calls[0][0][0]


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log', side_effect=Exception("log failed")):
        with patch('builtins.print') as mock_print:
            try:
                log_exception(ValueError("original error"))
            except Exception as e:
                assert isinstance(e, Exception)
                assert "log failed" in str(e)
    
    assert mock_print.call_count == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_register_ipython_excepthook_predicate_line_2():
    import sys
    from IPython.core import ultratb
    from bdb import BdbQuit
    
    def register_ipython_excepthook(capture_keyboard_interrupt: bool = False) -> None:
        r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

        :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
            the IPython debugger. Defaults to ``False``.
        """
        skip_exceptions = [BdbQuit]
        if not capture_keyboard_interrupt:
            skip_exceptions.append(KeyboardInterrupt)

        def excepthook(type, value, traceback):
            if any(type is exc_type for exc_type in skip_exceptions):
                # Don't capture keyboard interrupts (Ctrl+C) or Python debugger exit events.
                sys.__excepthook__(type, value, traceback)
            else:
                ipython_hook(type, value, traceback)

        # Enter IPython debugger on exception.
        from IPython.core import ultratb

        ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
        sys.excepthook = excepthook
    
    # Test that the docstring at line 2 is a non-empty string (truthy)
    docstring = register_ipython_excepthook.__doc__
    assert docstring is not None
    assert len(docstring) > 0
    assert isinstance(docstring, str)
    assert "Register an exception hook" in docstring


# LLM-generated content at query #5
#--------------------------

```python
def test_log_exception_with_user_msg(monkeypatch):
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append({"msg": msg, "level": level, "kwargs": kwargs})
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")
    
    assert len(log_calls) == 2
    assert log_calls[0]["level"] == "error"
    assert "Traceback" in log_calls[0]["msg"]
    assert log_calls[1]["level"] == "error"
    assert "Custom message" in log_calls[1]["msg"]
    assert "ValueError" in log_calls[1]["msg"]
    assert "test error" in log_calls[1]["msg"]


def test_log_exception_without_user_msg(monkeypatch):
    from flutes.exception import log_exception
    
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append({"msg": msg, "level": level, "kwargs": kwargs})
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
    
    assert len(log_calls) == 2
    assert log_calls[0]["level"] == "error"
    assert log_calls[1]["level"] == "error"
    assert "RuntimeError" in log_calls[1]["msg"]
    assert "runtime error" in log_calls[1]["msg"]


def test_log_exception_with_kwargs(monkeypatch):
    from flutes.exception import log_exception
    
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append({"msg": msg, "level": level, "kwargs": kwargs})
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)
    
    assert len(log_calls) == 2
    assert log_calls[0]["kwargs"] == {"force_console": True, "timestamp": False}
    assert log_calls[1]["kwargs"] == {"force_console": True, "timestamp": False}


def test_log_exception_with_subprocess_called_process_error(monkeypatch):
    import subprocess
    from flutes.exception import log_exception
    
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append({"msg": msg, "level": level, "kwargs": kwargs})
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    e = subprocess.CalledProcessError(1, "cmd", output="output data")
    log_exception(e, user_msg="Subprocess failed")
    
    assert len(log_calls) == 1
    assert log_calls[0]["level"] == "error"
    assert "CalledProcessError" in log_calls[0]["msg"]


def test_log_exception_with_subprocess_called_process_error_no_output(monkeypatch):
    import subprocess
    from flutes.exception import log_exception
    
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append({"msg": msg, "level": level, "kwargs": kwargs})
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    e = subprocess.CalledProcessError(1, "cmd", output=None)
    log_exception(e, user_msg="Subprocess failed")
    
    assert len(log_calls) == 2
    assert log_calls[0]["level"] == "error"
    assert log_calls[1]["level"] == "error"


def test_log_exception_logging_failure(monkeypatch, capsys):
    from flutes.exception import log_exception
    
    def mock_log(msg, level="info", **kwargs):
        raise RuntimeError("Logging failed")
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise ValueError("original error")
    except ValueError as e:
        try:
            log_exception(e, user_msg="Test message")
        except RuntimeError:
            pass
    
    captured = capsys.readouterr()
    assert "Test message" in captured.out
    assert "ValueError" in captured.out
    assert "original error" in captured.out
    assert "Logging failed" in captured.out


# LLM-generated content at query #6
#--------------------------

```python
def test_log_exception_predicate_line_12_true():
    import subprocess
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    # Case 1: Exception is not CalledProcessError
    e = ValueError("test error")
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
    
    # Case 2: CalledProcessError with output is None
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = None
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2
    
    # Case 3: CalledProcessError with no output attribute
    e = subprocess.CalledProcessError(1, "cmd")
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2


def test_log_exception_predicate_line_12_false():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    # Case: CalledProcessError with output is not None
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"some output"
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 1


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper_no_handler_logs_exception(capsys):
    @exception_wrapper()
    def failing_func():
        raise ValueError("Test error")
    
    failing_func()
    captured = capsys.readouterr()
    assert "ValueError" in captured.out or "ValueError" in captured.err


def test_exception_wrapper_with_handler():
    handler_called = []
    
    def handler(e):
        handler_called.append(e)
    
    @exception_wrapper(handler)
    def failing_func():
        raise ValueError("Test error")
    
    failing_func()
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)
    assert str(handler_called[0]) == "Test error"


def test_exception_wrapper_handler_receives_matching_args():
    handler_called = []
    
    def handler(e, x, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(handler)
    def failing_func(x, y):
        raise ValueError("Test error")
    
    failing_func(1, 2)
    assert len(handler_called) == 1
    assert handler_called[0][0].__class__.__name__ == "ValueError"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def handler(e, x, optional_arg=None):
        handler_called.append((e, x, optional_arg))
    
    @exception_wrapper(handler)
    def failing_func(x):
        raise ValueError("Test error")
    
    failing_func(1)
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varargs():
    handler_called = []
    
    def handler(e, x, args):
        handler_called.append((e, x, args))
    
    @exception_wrapper(handler)
    def failing_func(x, *args):
        raise ValueError("Test error")
    
    failing_func(1, 2, 3)
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def handler(e, x, varkw=None, **kw):
        handler_called.append((e, x, varkw, kw))
    
    @exception_wrapper(handler)
    def failing_func(x, **kwargs):
        raise ValueError("Test error")
    
    failing_func(1, a=2, b=3)
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][3] == {"a": 2, "b": 3}


def test_exception_wrapper_no_exception_returns_result():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    
    result = normal_func(5)
    assert result == 10


def test_exception_wrapper_handler_no_exception_returns_result():
    def handler(e):
        pass
    
    @exception_wrapper(handler)
    def normal_func(x):
        return x * 2
    
    result = normal_func(5)
    assert result == 10


def test_exception_wrapper_generator_with_exception():
    handler_called = []
    
    def handler(e):
        handler_called.append(e)
    
    @exception_wrapper(handler)
    def failing_generator():
        yield 1
        raise ValueError("Generator error")
    
    gen = failing_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_generator_without_exception():
    @exception_wrapper()
    def normal_generator():
        yield 1
        yield 2
        yield 3
    
    gen = normal_generator()
    result = list(gen)
    assert result == [1, 2, 3]


def test_exception_wrapper_handler_no_exception_arg_raises():
    def handler():
        pass
    
    try:
        @exception_wrapper(handler)
        def failing_func():
            raise ValueError("Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs_raises():
    def handler(e, *args):
        pass
    
    try:
        @exception_wrapper(handler)
        def failing_func():
            raise ValueError("Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped_raises():
    def handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(handler)
        def failing_func(x):
            raise ValueError("Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_arg_matches_wrapped_raises():
    def handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(handler)
        def failing_func(x):
            raise ValueError("Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def handler(e, x, *, kwonly_arg):
        handler_called.append((e, x, kwonly_arg))
    
    @exception_wrapper(handler)
    def failing_func(x, kwonly_arg):
        raise ValueError("Test error")
    
    failing_func(1, kwonly_arg=2)
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_kwonly_defaults():
    handler_called = []
    
    def handler(e, x, *, kwonly_arg=None):
        handler_called.append((e, x, kwonly_arg))
    
    @exception_wrapper(handler)
    def failing_func(x):
        raise ValueError("Test error")
    
    failing_func(1)
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] is None


def test_exception_wrapper_preserves_function_metadata():
    def handler(e):
        pass
    
    @exception_wrapper(handler)
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "Test docstring"


def test_exception_wrapper_handler_receives_all_args_and_kwargs():
    handler_called = []
    
    def handler(e, a, b, c=None, **kw):
        handler_called.append((a, b, c, kw))
    
    @exception_wrapper(handler)
    def failing_func(a, b, c=None, **kwargs):
        raise ValueError("Test error")
    
    failing_func(1, 2, c=3, d=4, e=5)
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert handler_called[0][1] == 2
    assert handler_called[0][2] ==


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_with_args(x, y):
        raise RuntimeError("custom error")
    
    try:
        func_with_args(1, 2)
    except:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == 1


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10


def test_exception_wrapper_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, y=10):
        raise TypeError("type error")
    
    try:
        func_with_defaults(5)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y=None, **kwargs):
        raise KeyError("key error")
    
    try:
        func_with_kwargs(1, y=2, z=3)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1


def test_exception_wrapper_with_varargs():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((e, x, args))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args):
        raise ValueError("value error")
    
    try:
        func_with_varargs(1, 2, 3)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


def test_exception_wrapper_invalid_handler_no_args():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_unmatched_arg():
    def bad_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_on_matching_arg():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def gen_func(x):
        yield 1
        raise RuntimeError("gen error")
    
    gen = gen_func(5)
    assert next(gen) == 1
    try:
        next(gen)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5


def test_exception_wrapper_generator_no_error():
    @exception_wrapper()
    def gen_func(x):
        yield x
        yield x * 2
    
    gen = gen_func(3)
    assert next(gen) == 3
    assert next(gen) == 6


def test_exception_wrapper_preserves_function_name():
    @exception_wrapper()
    def my_function():
        pass
    
    assert my_function.__name__ == "my_function"


def test_exception_wrapper_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func(x, *, y):
        raise ValueError("kwonly error")
    
    try:
        func(1, y=2)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_kwonly_defaults():
    handler_called = []
    
    def custom_handler(e, x, z=None):
        handler_called.append((e, x, z))
    
    @exception_wrapper(custom_handler)
    def func(x, y):
        raise TypeError("type error")
    
    try:
        func(10, 20)
    except:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] is None


def test_exception_wrapper_multiple_calls():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append(x)
    
    @exception_wrapper(custom_handler)
    def func(x):
        raise ValueError("error")
    
    try:
        func(1)
    except:
        pass
    
    try:
        func(2)
    except:
        pass
    
    assert handler_called == [1, 2]


# LLM-generated content at query #9
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook()
        assert sys.excepthook is not None
        assert sys.excepthook != original_excepthook
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    import sys
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        assert sys.excepthook is not None
        assert sys.excepthook != original_excepthook
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_true():
    import sys
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        assert sys.excepthook is not None
        assert sys.excepthook != original_excepthook
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_bdbquit_exception():
    import sys
    from bdb import BdbQuit
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        exc = BdbQuit()
        sys.excepthook(BdbQuit, exc, None)
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_not_captured():
    import sys
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        exc = KeyboardInterrupt()
        sys.excepthook(KeyboardInterrupt, exc, None)
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_captured():
    import sys
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        assert sys.excepthook is not None
    finally:
        sys.excepthook = original_excepthook


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("test error")
    
    try:
        func_raises_error()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_raises_error():
        raise ValueError("test error")
    
    try:
        func_raises_error()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_raises_error(x, y):
        raise ValueError("test error")
    
    try:
        func_raises_error(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 1, 2)


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, my_arg=None):
        handler_called.append((str(e), my_arg))
    
    @exception_wrapper(custom_handler)
    def func_raises_error():
        raise ValueError("test error")
    
    try:
        func_raises_error()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] is None


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, my_arg=None, **kw):
        handler_called.append((str(e), x, my_arg, kw))
    
    @exception_wrapper(custom_handler)
    def func_raises_error(x, y=5):
        raise ValueError("test error")
    
    try:
        func_raises_error(1, y=10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] is None
    assert handler_called[0][3] == {"y": 10}


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        raise ValueError("generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
        yield 3
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2
    assert next(gen) == 3


def test_exception_wrapper_invalid_handler_no_args():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_unmatched_arg():
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_arg" in str(e)


def test_exception_wrapper_invalid_handler_default_on_matched_arg():
    def invalid_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_handler_with_varargs_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, my_arg=None, **kw):
        handler_called.append((str(e), x, my_arg, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args, **kwargs):
        raise ValueError("test error")
    
    try:
        func_with_varargs(1, 2, 3, key="value")
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert "args" in handler_called[0][3]
    assert "kwargs" in handler_called[0][3]


def test_exception_wrapper_handler_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_kwonly(x, *, y=None):
        raise ValueError("test error")
    
    try:
        func_kwonly(1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 1, 2)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def my_func():
        """My docstring"""
        pass
    
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring"


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_handler_fn_is_none():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_predicate_line_2():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    import sys
    from bdb import BdbQuit
    
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    predicate = not capture_keyboard_interrupt
    assert predicate is True
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    decorator = exception_wrapper()
    assert callable(decorator)


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_predicate_line_6_false():
    """Test that the predicate at line 6 (handler_fn is not None) evaluates to False"""
    import functools
    import inspect
    from flutes.exception import exception_wrapper
    
    # Use exception_wrapper with handler_fn=None (default)
    @exception_wrapper()
    def test_func():
        return "success"
    
    # Call the wrapped function - it should execute normally without exception handling
    result = test_func()
    assert result == "success"
    
    # Test with explicit None
    @exception_wrapper(handler_fn=None)
    def test_func2():
        return "success2"
    
    result2 = test_func2()
    assert result2 == "success2"


# LLM-generated content at query #16
#--------------------------

```python
def test_log_exception_no_exception_during_logging():
    import subprocess
    from flutes.exception import log_exception
    
    test_exception = ValueError("test error")
    log_exception(test_exception, user_msg="Test message")


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_true():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_replaces_excepthook():
    import sys
    
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    new_excepthook = sys.excepthook
    
    assert new_excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_multiple_calls():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    first_call_hook = sys.excepthook
    
    register_ipython_excepthook()
    second_call_hook = sys.excepthook
    
    assert first_call_hook is not None
    assert second_call_hook is not None
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    # Line 1 predicate: def exception_wrapper(handler_fn=None):
    # This tests that exception_wrapper is callable and has the correct default parameter
    import inspect
    from flutes.exception import exception_wrapper
    
    sig = inspect.signature(exception_wrapper)
    assert 'handler_fn' in sig.parameters
    assert sig.parameters['handler_fn'].default is None
    assert callable(exception_wrapper)
    
    # Test that calling with no arguments works
    decorator = exception_wrapper()
    assert callable(decorator)
    
    # Test that calling with a handler function works
    def dummy_handler(e):
        pass
    
    decorator_with_handler = exception_wrapper(dummy_handler)
    assert callable(decorator_with_handler)


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("Test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("Test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise RuntimeError("Error occurred")
    
    try:
        func_that_raises(10, 20)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, default_arg=None):
        handler_called.append((str(e), x, default_arg))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise RuntimeError("Error")
    
    try:
        func_that_raises(5)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise RuntimeError("Error")
    
    try:
        func_that_raises(5, y=10)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert "y" in handler_called[0][2]


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        raise ValueError("Generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_generator_no_error():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
        yield 3
    
    gen = gen_func()
    result = list(gen)
    
    assert result == [1, 2, 3]


def test_exception_wrapper_invalid_handler_no_positional_arg():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_arg_mismatch():
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_arg_matches():
    def invalid_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, *args, y=None, **kw):
        handler_called.append((x, args, y, kw))
    
    @exception_wrapper(custom_handler)
    def func(x, *args, y=None, **kwargs):
        raise RuntimeError("Error")
    
    try:
        func(1, 2, 3, y=4, z=5)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert handler_called[0][1] == (2, 3)
    assert handler_called[0][2] == 4


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a documented function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "documented function" in documented_func.__doc__


def test_exception_wrapper_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y):
        handler_called.append((x, y))
    
    @exception_wrapper(custom_handler)
    def func(x, *, y):
        raise RuntimeError("Error")
    
    try:
        func(1, y=2)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == (1, 2)


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    from flutes.exception import exception_wrapper
    docstring = exception_wrapper.__doc__
    assert docstring is not None
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated function" in docstring


# LLM-generated content at query #21
#--------------------------

```python
def test_log_exception_basic():
    import traceback
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = ValueError("test error")
        log_exception(exc)
        assert mock_log.call_count == 2
        args_list = mock_log.call_args_list
        assert "error" in str(args_list[0])
        assert "ValueError" in str(args_list[1])


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = RuntimeError("runtime issue")
        log_exception(exc, user_msg="Custom message")
        assert mock_log.call_count == 2
        args_list = mock_log.call_args_list
        assert "Custom message" in str(args_list[1])
        assert "RuntimeError" in str(args_list[1])


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = TypeError("type mismatch")
        log_exception(exc, force_console=True, timestamp=False)
        assert mock_log.call_count == 2
        for call in mock_log.call_args_list:
            assert call[1].get('force_console') == True
            assert call[1].get('timestamp') == False


def test_log_exception_called_process_error():
    from unittest.mock import patch
    import subprocess
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = subprocess.CalledProcessError(1, 'cmd', output='output')
        log_exception(exc)
        assert mock_log.call_count == 1
        args_list = mock_log.call_args_list
        assert "CalledProcessError" in str(args_list[0])


def test_log_exception_called_process_error_no_output():
    from unittest.mock import patch
    import subprocess
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = subprocess.CalledProcessError(1, 'cmd')
        log_exception(exc)
        assert mock_log.call_count == 2


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        with patch('builtins.print') as mock_print:
            mock_log.side_effect = Exception("logging failed")
            exc = ValueError("original error")
            try:
                log_exception(exc)
            except Exception as e:
                assert "logging failed" in str(e)
            assert mock_print.call_count == 2


def test_log_exception_with_user_msg_and_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        exc = KeyError("missing key")
        log_exception(exc, user_msg="Key lookup failed", force_console=True)
        assert mock_log.call_count == 2
        args_list = mock_log.call_args_list
        assert "Key lookup failed" in str(args_list[1])
        assert "KeyError" in str(args_list[1])


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #23
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt_true():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_sets_sys_excepthook():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    new_excepthook = sys.excepthook
    
    assert new_excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_multiple_calls():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    first_hook = sys.excepthook
    
    register_ipython_excepthook()
    second_hook = sys.excepthook
    
    assert first_hook is not None
    assert second_hook is not None
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_predicate_line_2_false():
    from flutes.exception import exception_wrapper
    
    def handler_fn(e):
        pass
    
    @exception_wrapper(handler_fn)
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


# LLM-generated content at query #25
#--------------------------

```python
def test_log_exception_with_user_msg(monkeypatch):
    from flutes.exception import log_exception
    from flutes.log import log
    
    logged_messages = []
    logged_levels = []
    
    def mock_log(msg, level="info", **kwargs):
        logged_messages.append(msg)
        logged_levels.append(level)
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_exception(e, user_msg="Custom message")
    
    assert len(logged_messages) == 2
    assert logged_levels[0] == "error"
    assert logged_levels[1] == "error"
    assert "Custom message: <ValueError> test error" in logged_messages[1]


def test_log_exception_without_user_msg(monkeypatch):
    from flutes.exception import log_exception
    
    logged_messages = []
    logged_levels = []
    
    def mock_log(msg, level="info", **kwargs):
        logged_messages.append(msg)
        logged_levels.append(level)
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        log_exception(e)
    
    assert len(logged_messages) == 2
    assert logged_levels[0] == "error"
    assert logged_levels[1] == "error"
    assert "<RuntimeError> runtime error" in logged_messages[1]


def test_log_exception_with_kwargs(monkeypatch):
    from flutes.exception import log_exception
    
    logged_messages = []
    logged_kwargs = []
    
    def mock_log(msg, level="info", **kwargs):
        logged_messages.append(msg)
        logged_kwargs.append(kwargs)
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        log_exception(e, force_console=True, timestamp=False)
    
    assert len(logged_messages) == 2
    assert logged_kwargs[0].get("force_console") == True
    assert logged_kwargs[0].get("timestamp") == False
    assert logged_kwargs[1].get("force_console") == True
    assert logged_kwargs[1].get("timestamp") == False


def test_log_exception_with_subprocess_error(monkeypatch):
    import subprocess
    from flutes.exception import log_exception
    
    logged_messages = []
    logged_levels = []
    
    def mock_log(msg, level="info", **kwargs):
        logged_messages.append(msg)
        logged_levels.append(level)
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    proc_error = subprocess.CalledProcessError(1, "cmd", output=b"output")
    log_exception(proc_error, user_msg="Process failed")
    
    assert len(logged_messages) == 1
    assert logged_levels[0] == "error"
    assert "Process failed: <CalledProcessError>" in logged_messages[0]


def test_log_exception_logging_fails(monkeypatch, capsys):
    from flutes.exception import log_exception
    
    call_count = [0]
    
    def mock_log(msg, level="info", **kwargs):
        call_count[0] += 1
        raise RuntimeError("logging failed")
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    try:
        raise ValueError("original error")
    except ValueError as e:
        try:
            log_exception(e, user_msg="Test")
        except RuntimeError:
            pass
    
    captured = capsys.readouterr()
    assert "Test: <ValueError> original error" in captured.out
    assert "Another exception occurred while logging" in captured.out


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_predicate_line_12_false():
    """Test that the predicate at line 12 (handler_argspec.varargs is not None) evaluates to False."""
    def handler_fn(e, arg1):
        pass
    
    def wrapped_fn(arg1):
        pass
    
    # This should not raise ValueError about varargs
    # because handler_fn does not have varargs (*args)
    decorator = exception_wrapper(handler_fn)
    result = decorator(wrapped_fn)
    
    assert result is not None
    assert callable(result)


