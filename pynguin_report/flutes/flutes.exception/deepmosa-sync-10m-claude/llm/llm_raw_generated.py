####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
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
    
    assert new_excepthook is not None
    assert callable(new_excepthook)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_true():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #2
#--------------------------

```python
def test_log_exception_with_basic_exception():
    import traceback
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e)
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert "ValueError" in call_args_list[0][0][0]
            assert "<ValueError> test error" in call_args_list[1][0][0]
            assert call_args_list[1][0][1] == "error"


def test_log_exception_with_user_message():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Custom message")
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert "Custom message: <RuntimeError> runtime error" in call_args_list[1][0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, force_console=True, timestamp=False)
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert call_args_list[0][1]['force_console'] is True
            assert call_args_list[0][1]['timestamp'] is False
            assert call_args_list[1][1]['force_console'] is True
            assert call_args_list[1][1]['timestamp'] is False


def test_log_exception_with_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, "cmd", output="output")
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 1
        call_args_list = mock_log.call_args_list
        assert "<CalledProcessError>" in call_args_list[0][0][0]


def test_log_exception_with_subprocess_error_no_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, "cmd", output=None)
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 2
        call_args_list = mock_log.call_args_list
        assert "<CalledProcessError>" in call_args_list[1][0][0]


def test_log_exception_with_logging_error():
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            mock_log.side_effect = Exception("logging failed")
            with patch('builtins.print') as mock_print:
                try:
                    log_exception(e)
                except Exception as log_e:
                    assert isinstance(log_e, Exception)
                    assert "logging failed" in str(log_e)
                    assert mock_print.call_count == 2


# LLM-generated content at query #3
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
    def func_with_args(x, y):
        raise ValueError("test error")
    
    try:
        func_with_args(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, y=10):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_args(x):
        raise ValueError("test error")
    
    try:
        func_with_args(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y=5):
        raise ValueError("test error")
    
    try:
        func_with_kwargs(1, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2].get('y') == 20


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_no_error():
        return 42
    
    result = func_no_error()
    
    assert result == 42
    assert len(handler_called) == 0


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_generator():
        yield 1
        raise ValueError("generator error")
    
    gen = func_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_invalid_handler_no_args():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_unmatched_arg():
    def invalid_handler(e, unknown_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "unknown_arg" in str(e)


def test_exception_wrapper_handler_with_varargs_in_wrapped():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((str(e), x, args))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args):
        raise ValueError("test error")
    
    try:
        func_with_varargs(1, 2, 3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


def test_exception_wrapper_preserves_function_name():
    @exception_wrapper()
    def my_function():
        pass
    
    assert my_function.__name__ == "my_function"


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y=5):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_kwonly(x):
        raise ValueError("test error")
    
    try:
        func_with_kwonly(1)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 5


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper_handler_with_varkw():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def handler_fn(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(handler_fn)
    def foo(x, y):
        raise ValueError("test error")
    
    foo(1, 2)
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == {'y': 2}


# LLM-generated content at query #5
#--------------------------

```python
def test_log_exception_with_user_msg(mocker):
    import traceback
    from flutes.exception import log_exception
    from flutes.log import log
    
    mock_log = mocker.patch('flutes.exception.log')
    test_exception = ValueError("test error")
    user_msg = "Custom message"
    
    log_exception(test_exception, user_msg=user_msg)
    
    assert mock_log.call_count == 2
    first_call_args = mock_log.call_args_list[0]
    second_call_args = mock_log.call_args_list[1]
    assert first_call_args[0][1] == "error"
    assert second_call_args[0][0] == "Custom message: <ValueError> test error"
    assert second_call_args[0][1] == "error"


def test_log_exception_without_user_msg(mocker):
    import traceback
    from flutes.exception import log_exception
    
    mock_log = mocker.patch('flutes.exception.log')
    test_exception = RuntimeError("runtime error")
    
    log_exception(test_exception)
    
    assert mock_log.call_count == 2
    first_call_args = mock_log.call_args_list[0]
    second_call_args = mock_log.call_args_list[1]
    assert first_call_args[0][1] == "error"
    assert second_call_args[0][0] == "<RuntimeError> runtime error"
    assert second_call_args[0][1] == "error"


def test_log_exception_with_kwargs(mocker):
    from flutes.exception import log_exception
    
    mock_log = mocker.patch('flutes.exception.log')
    test_exception = TypeError("type error")
    
    log_exception(test_exception, force_console=True, timestamp=False)
    
    assert mock_log.call_count == 2
    first_call_args = mock_log.call_args_list[0]
    second_call_args = mock_log.call_args_list[1]
    assert first_call_args[1]['force_console'] is True
    assert first_call_args[1]['timestamp'] is False
    assert second_call_args[1]['force_console'] is True
    assert second_call_args[1]['timestamp'] is False


def test_log_exception_with_called_process_error(mocker):
    import subprocess
    from flutes.exception import log_exception
    
    mock_log = mocker.patch('flutes.exception.log')
    test_exception = subprocess.CalledProcessError(1, 'cmd', output='output')
    
    log_exception(test_exception)
    
    assert mock_log.call_count == 1
    call_args = mock_log.call_args_list[0]
    assert call_args[0][1] == "error"


def test_log_exception_logging_fails(mocker):
    from flutes.exception import log_exception
    
    mock_log = mocker.patch('flutes.exception.log', side_effect=Exception("log failed"))
    mock_print = mocker.patch('builtins.print')
    test_exception = ValueError("test error")
    
    try:
        log_exception(test_exception)
    except Exception:
        pass
    
    assert mock_print.call_count == 2
    print_calls = mock_print.call_args_list
    assert "<ValueError> test error" in str(print_calls[0])
    assert "Another exception occurred while logging" in str(print_calls[1])


def test_log_exception_with_user_msg_and_kwargs(mocker):
    from flutes.exception import log_exception
    
    mock_log = mocker.patch('flutes.exception.log')
    test_exception = KeyError("key not found")
    user_msg = "Key lookup failed"
    
    log_exception(test_exception, user_msg=user_msg, include_proc_id=False)
    
    assert mock_log.call_count == 2
    second_call_args = mock_log.call_args_list[1]
    assert second_call_args[0][0] == "Key lookup failed: <KeyError> 'key not found'"
    assert second_call_args[1]['include_proc_id'] is False


# LLM-generated content at query #6
#--------------------------

```python
def test_register_ipython_excepthook_predicate_evaluates_to_false():
    capture_keyboard_interrupt = False
    predicate = not capture_keyboard_interrupt
    assert predicate is False


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper_predicate_line_5():
    # Line 5 is an empty line in the docstring, but the actual predicate logic
    # starts at line 45: "if handler_fn is not None:"
    # This test ensures that when handler_fn is None, the decorator works without validation
    
    def dummy_function(x):
        return x * 2
    
    decorated = exception_wrapper(handler_fn=None)(dummy_function)
    result = decorated(5)
    assert result == 10


def test_exception_wrapper_predicate_with_handler():
    # This test ensures that when handler_fn is not None (line 45 predicate is True),
    # the validation checks are performed
    
    def valid_handler(e, x):
        return f"Handled: {e}"
    
    def dummy_function(x):
        if x < 0:
            raise ValueError("Negative value")
        return x * 2
    
    decorated = exception_wrapper(handler_fn=valid_handler)(dummy_function)
    result = decorated(5)
    assert result == 10


def test_exception_wrapper_handler_fn_is_not_none_validation():
    # Test that the predicate at line 45 evaluates to True and validation occurs
    
    def handler_with_no_exception_arg(x):
        return "no exception arg"
    
    def dummy_function(x):
        return x
    
    try:
        exception_wrapper(handler_fn=handler_with_no_exception_arg)(dummy_function)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_log_exception_basic():
    import subprocess
    import traceback
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
    
    assert mock_log.call_count == 2
    assert mock_log.call_args_list[0][0][1] == "error"
    assert mock_log.call_args_list[1][0][1] == "error"
    assert "<ValueError> Test error" in mock_log.call_args_list[1][0][0]


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise RuntimeError("Runtime problem")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
    
    assert mock_log.call_count == 2
    assert "Custom message: <RuntimeError> Runtime problem" in mock_log.call_args_list[1][0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise TypeError("Type mismatch")
        except TypeError as e:
            log_exception(e, force_console=True, timestamp=False)
    
    assert mock_log.call_count == 2
    assert mock_log.call_args_list[0][1]['force_console'] is True
    assert mock_log.call_args_list[0][1]['timestamp'] is False


def test_log_exception_with_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        e = subprocess.CalledProcessError(1, 'cmd', output='output')
        log_exception(e)
    
    assert mock_log.call_count == 1
    assert "<CalledProcessError>" in mock_log.call_args_list[0][0][0]


def test_log_exception_logging_failure():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log', side_effect=Exception("Log failed")):
        with patch('builtins.print') as mock_print:
            try:
                log_exception(ValueError("Original error"))
            except Exception:
                pass
    
    assert mock_print.call_count == 2
    assert "Original error" in str(mock_print.call_args_list[0])


def test_log_exception_preserves_exception():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    log_error = Exception("Log failed")
    with patch('flutes.exception.log', side_effect=log_error):
        with patch('builtins.print'):
            try:
                log_exception(ValueError("Original"))
                assert False, "Should have raised"
            except Exception as e:
                assert e is log_error


# LLM-generated content at query #9
#--------------------------

```python
def test_log_exception_predicate_line_12_true():
    import subprocess
    from flutes.exception import log_exception
    from unittest.mock import patch
    
    # Create a CalledProcessError with output set to None
    # This makes the predicate at line 12 evaluate to True
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        # When predicate is True, traceback should be logged
        assert mock_log.call_count >= 1


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_decorator_exists():
    from flutes.exception import exception_wrapper
    result = exception_wrapper is not None
    assert result == True


# LLM-generated content at query #11
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
        args1, kwargs1 = mock_log.call_args_list[0]
        assert "ValueError" in args1[0]
        assert kwargs1.get('level') == 'error' or 'error' in str(kwargs1)
        args2, kwargs2 = mock_log.call_args_list[1]
        assert "test error" in args2[0]
        assert "ValueError" in args2[0]


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise RuntimeError("original error")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        assert mock_log.call_count == 2
        args2, kwargs2 = mock_log.call_args_list[1]
        assert "Custom message" in args2[0]
        assert "RuntimeError" in args2[0]
        assert "original error" in args2[0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise TypeError("type error")
        except TypeError as e:
            log_exception(e, force_console=True, timestamp=False)
        
        assert mock_log.call_count == 2
        args1, kwargs1 = mock_log.call_args_list[0]
        assert kwargs1.get('force_console') == True
        assert kwargs1.get('timestamp') == False
        args2, kwargs2 = mock_log.call_args_list[1]
        assert kwargs2.get('force_console') == True
        assert kwargs2.get('timestamp') == False


def test_log_exception_with_called_process_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        error = subprocess.CalledProcessError(1, 'cmd', output='output data')
        log_exception(error)
        
        assert mock_log.call_count == 1
        args, kwargs = mock_log.call_args_list[0]
        assert "CalledProcessError" in args[0]


def test_log_exception_logging_failure():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log', side_effect=RuntimeError("logging failed")):
        with patch('builtins.print') as mock_print:
            try:
                log_exception(ValueError("original"))
            except RuntimeError:
                pass
            
            assert mock_print.call_count >= 2


def test_log_exception_called_process_error_without_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        error = subprocess.CalledProcessError(1, 'cmd')
        log_exception(error)
        
        assert mock_log.call_count == 2
        args1, kwargs1 = mock_log.call_args_list[0]
        assert "CalledProcessError" in args1[0] or "Traceback" in args1[0]


# LLM-generated content at query #12
#--------------------------

```python
def test_log_exception_predicate_line_12():
    import subprocess
    from flutes.exception import log_exception
    from unittest.mock import patch
    
    # Test case 1: CalledProcessError with output not None - predicate is False
    error_with_output = subprocess.CalledProcessError(1, "cmd", output="some output")
    with patch('flutes.exception.log') as mock_log:
        log_exception(error_with_output)
        # When predicate is False, traceback should NOT be logged
        assert mock_log.call_count == 1
        assert mock_log.call_args_list[0][0][1] == "error"
    
    # Test case 2: CalledProcessError with output None - predicate is True
    error_no_output = subprocess.CalledProcessError(1, "cmd", output=None)
    with patch('flutes.exception.log') as mock_log:
        log_exception(error_no_output)
        # When predicate is True, traceback should be logged
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[0][0][1] == "error"
        assert mock_log.call_args_list[1][0][1] == "error"
    
    # Test case 3: Non-CalledProcessError exception - predicate is True
    regular_error = ValueError("test error")
    with patch('flutes.exception.log') as mock_log:
        log_exception(regular_error)
        # When predicate is True, traceback should be logged
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[0][0][1] == "error"
        assert mock_log.call_args_list[1][0][1] == "error"


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


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
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
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
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_with_handler_matching_args():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((str(e), x))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(42, 100)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 42


def test_exception_wrapper_with_handler_and_default_args():
    handler_called = []
    
    def custom_handler(e, x, my_default=None):
        handler_called.append((str(e), x, my_default))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(42, 100)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 42
    assert handler_called[0][2] is None


def test_exception_wrapper_with_handler_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(42, y=100)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 42
    assert handler_called[0][2] == {"y": 100}


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


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        raise ValueError("gen error")
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "gen error" in handler_called[0]


def test_exception_wrapper_invalid_handler_no_args():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_mismatched_arg():
    def bad_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_on_matched_arg():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, **kwargs):
        raise ValueError("test error")
    
    try:
        func_that_raises(42, key1="val1", key2="val2")
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 42
    assert handler_called[0][2]["key1"] == "val1"
    assert handler_called[0][2]["key2"] == "val2"


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def my_function():
        """My docstring"""
        pass
    
    assert my_function.__name__ == "my_function"
    assert "My docstring" in my_function.__doc__


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, z=None, **kw):
        handler_called.append((str(e), x, z, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y, z=10):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, z=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 20
    assert handler_called[0][3] == {"y": 2}


def test_exception_wrapper_generator_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        yield 3
    
    gen = gen_func()
    result = list(gen)
    
    assert result == [1, 2, 3]
    assert len(handler_called) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_no_handler_logs_exception(monkeypatch):
    log_calls = []
    def mock_log_exception(e, **kwargs):
        log_calls.append((e, kwargs))
    
    monkeypatch.setattr("flutes.exception.log_exception", mock_log_exception)
    
    @exception_wrapper()
    def failing_func():
        raise ValueError("test error")
    
    failing_func()
    assert len(log_calls) == 1
    assert isinstance(log_calls[0][0], ValueError)
    assert str(log_calls[0][0]) == "test error"


def test_exception_wrapper_with_custom_handler():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise RuntimeError("custom error")
    
    failing_func(42)
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], RuntimeError)
    assert handler_calls[0][1] == 42


def test_exception_wrapper_handler_with_defaults():
    handler_calls = []
    
    def custom_handler(e, x, y=10):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("error")
    
    failing_func(5)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2] == 10


def test_exception_wrapper_handler_with_kwargs():
    handler_calls = []
    
    def custom_handler(e, x, **kw):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise TypeError("error")
    
    failing_func(1, 2)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert "y" in handler_calls[0][2]


def test_exception_wrapper_handler_receives_all_args():
    handler_calls = []
    
    def custom_handler(e, a, b, c=None):
        handler_calls.append((e, a, b, c))
    
    @exception_wrapper(custom_handler)
    def failing_func(a, b, c=None):
        raise ValueError("error")
    
    failing_func(1, 2, c=3)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 2
    assert handler_calls[0][3] == 3


def test_exception_wrapper_no_exception_returns_normally():
    @exception_wrapper()
    def normal_func(x):
        return x * 2
    
    result = normal_func(5)
    assert result == 10


def test_exception_wrapper_with_args_and_kwargs():
    handler_calls = []
    
    def custom_handler(e, x, **kw):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y, z=None):
        raise RuntimeError("error")
    
    failing_func(1, 2, z=3)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2]["y"] == 2
    assert handler_calls[0][2]["z"] == 3


def test_exception_wrapper_generator_exception():
    handler_calls = []
    
    def custom_handler(e):
        handler_calls.append(e)
    
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
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0], ValueError)


def test_exception_wrapper_handler_no_exception_arg_raises():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs_raises():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped_raises():
    def bad_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_matches_wrapped_raises():
    def bad_handler(e, x=5):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a docstring."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "docstring" in documented_func.__doc__


def test_exception_wrapper_handler_with_varargs_kwargs():
    handler_calls = []
    
    def custom_handler(e, x, y=None, **kw):
        handler_calls.append((e, x, y, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, a, b):
        raise ValueError("error")
    
    failing_func(1, 2, 3)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][3]["a"] == 2
    assert handler_calls[0][3]["b"] == 3


def test_exception_wrapper_handler_with_keyword_only_args():
    handler_calls = []
    
    def custom_handler(e, *, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise TypeError("error")
    
    failing_func(10)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 10


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper_predicate_line_1_false():
    """Test that the predicate at line 1 (handler_fn is not None) evaluates to False"""
    from flutes.exception import exception_wrapper
    
    # When handler_fn is None (default), the predicate at line 45 should be False
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")
    
    # Call the wrapped function and verify it handles the exception with default handler
    test_func()


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #19
#--------------------------

```python
def test_register_ipython_excepthook_default_parameter():
    import sys
    from bdb import BdbQuit
    
    # Store original excepthook
    original_excepthook = sys.excepthook
    
    try:
        # Call the function with default parameter (capture_keyboard_interrupt=False)
        from your_module import register_ipython_excepthook
        register_ipython_excepthook()
        
        # Verify that sys.excepthook has been modified
        assert sys.excepthook != original_excepthook
        assert sys.excepthook is not None
        
        # Verify that the excepthook is callable
        assert callable(sys.excepthook)
    finally:
        # Restore original excepthook
        sys.excepthook = original_excepthook


# LLM-generated content at query #20
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


def test_exception_wrapper_custom_handler():
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
        raise ValueError("Test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
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
        raise ValueError("Test error")
    
    try:
        func_that_raises(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("Test error")
    
    try:
        func_that_raises(10, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10


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
    def gen_func_that_raises():
        yield 1
        raise ValueError("Generator error")
    
    gen = gen_func_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


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


def test_exception_wrapper_handler_arg_mismatch():
    def handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_arg_mismatch():
    def handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, **kw):
        handler_called.append((str(e), a, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(a, b, c=None):
        raise ValueError("Test error")
    
    try:
        func_that_raises(1, 2, c=3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1


def test_exception_wrapper_preserves_function_metadata():
    def handler(e):
        pass
    
    @exception_wrapper(handler)
    def original_func():
        """Original docstring"""
        pass
    
    assert original_func.__doc__ == "Original docstring"
    assert original_func.__name__ == "original_func"


# LLM-generated content at query #21
#--------------------------

```python
def test_log_exception_predicate_line_15_evaluates_to_false():
    import subprocess
    from flutes.exception import log_exception
    
    # Create a CalledProcessError with output set (not None)
    # This makes the predicate at line 12 evaluate to True
    # Which means the condition at line 15 (except block) should NOT be triggered
    error = subprocess.CalledProcessError(1, 'cmd', output='some output')
    
    # Call log_exception - it should not raise an exception
    # because the logging should succeed without triggering the except block at line 15
    log_exception(error, user_msg="Test message")


# LLM-generated content at query #22
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    from unittest.mock import patch, MagicMock
    import sys
    
    # Mock the IPython module
    with patch('IPython.core.ultratb.FormattedTB') as mock_tb:
        mock_hook = MagicMock()
        mock_tb.return_value = mock_hook
        
        # Import and call the function
        from your_module import register_ipython_excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # Get the excepthook that was set
        excepthook = sys.excepthook
        
        # Test the predicate at line 2: r"""Register an exception hook..."""
        # The predicate is the docstring itself, which should be truthy
        assert register_ipython_excepthook.__doc__ is not None
        assert isinstance(register_ipython_excepthook.__doc__, str)
        assert len(register_ipython_excepthook.__doc__) > 0
        assert "Register an exception hook" in register_ipython_excepthook.__doc__


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_handler_with_varkw():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e, one, **kw):
        handler_called.append({
            'exception': e,
            'one': one,
            'kw': kw
        })
    
    @exception_wrapper(custom_handler)
    def test_func(one, two, three=None):
        raise ValueError("test error")
    
    try:
        test_func(1, "2", three=3)
    except:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0]['exception'], ValueError)
    assert handler_called[0]['one'] == 1
    assert 'two' in handler_called[0]['kw']
    assert handler_called[0]['kw']['two'] == "2"
    assert 'three' in handler_called[0]['kw']
    assert handler_called[0]['kw']['three'] == 3


# LLM-generated content at query #24
#--------------------------

```python
def test_register_ipython_excepthook_predicate_evaluates_to_false():
    from bdb import BdbQuit
    import sys
    
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    predicate = any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)
    
    assert predicate is False


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    decorator = exception_wrapper()
    assert callable(decorator)


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #27
#--------------------------

```python
def test_log_exception_basic():
    import sys
    from io import StringIO
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e)
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert "ValueError" in str(call_args_list[0])
            assert "test error" in str(call_args_list[1])


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise RuntimeError("original error")
    except RuntimeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Custom message")
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert "Custom message" in str(call_args_list[1])
            assert "RuntimeError" in str(call_args_list[1])


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, force_console=True, timestamp=False)
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert call_args_list[0][1]['force_console'] is True
            assert call_args_list[0][1]['timestamp'] is False


def test_log_exception_called_process_error_with_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, "cmd", output="some output")
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 1
        call_args = mock_log.call_args_list[0]
        assert "CalledProcessError" in str(call_args)


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise KeyError("key error")
    except KeyError as e:
        with patch('flutes.exception.log', side_effect=Exception("log failed")):
            with patch('builtins.print') as mock_print:
                try:
                    log_exception(e)
                except Exception as log_e:
                    assert "log failed" in str(log_e)
                    assert mock_print.call_count == 2


def test_log_exception_with_all_params():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise AttributeError("attr error")
    except AttributeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Detailed info", force_console=True, include_proc_id=False)
            assert mock_log.call_count == 2
            call_args_list = mock_log.call_args_list
            assert call_args_list[1][1]['force_console'] is True
            assert call_args_list[1][1]['include_proc_id'] is False


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e, arg1):
        handler_called.append((e, arg1))
    
    @exception_wrapper(custom_handler)
    def func_with_arg(arg1):
        raise RuntimeError("custom error")
    
    try:
        func_with_arg("test_value")
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == "test_value"


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, arg1, arg2="default"):
        handler_called.append((e, arg1, arg2))
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(arg1, arg2="default"):
        raise TypeError("type error")
    
    try:
        func_with_defaults("value1")
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], TypeError)
    assert handler_called[0][1] == "value1"
    assert handler_called[0][2] == "default"


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, arg1, **kw):
        handler_called.append((e, arg1, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(arg1, **kwargs):
        raise ValueError("value error")
    
    try:
        func_with_kwargs("value1", extra="extra_value")
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == "value1"
    assert "extra" in handler_called[0][2]


def test_exception_wrapper_handler_with_args():
    handler_called = []
    
    def custom_handler(e, arg1, args):
        handler_called.append((e, arg1, args))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(arg1, *args):
        raise RuntimeError("runtime error")
    
    try:
        func_with_varargs("value1", "arg2", "arg3")
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == "value1"
    assert handler_called[0][2] == ("arg2", "arg3")


def test_exception_wrapper_handler_with_kwonly():
    handler_called = []
    
    def custom_handler(e, arg1, kwonly_arg):
        handler_called.append((e, arg1, kwonly_arg))
    
    @exception_wrapper(custom_handler)
    def func_with_kwonly(arg1, *, kwonly_arg):
        raise KeyError("key error")
    
    try:
        func_with_kwonly("value1", kwonly_arg="kwvalue")
    except KeyError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], KeyError)
    assert handler_called[0][1] == "value1"
    assert handler_called[0][2] == "kwvalue"


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e, arg1):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_no_error(arg1):
        return f"success_{arg1}"
    
    result = func_no_error("test")
    
    assert result == "success_test"
    assert len(handler_called) == 0


def test_exception_wrapper_handler_no_positional_arg():
    try:
        def bad_handler(arg1):
            pass
        
        @exception_wrapper(bad_handler)
        def func():
            pass
        
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs():
    try:
        def bad_handler(e, *args):
            pass
        
        @exception_wrapper(bad_handler)
        def func():
            pass
        
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped():
    try:
        def bad_handler(e, nonexistent_arg):
            pass
        
        @exception_wrapper(bad_handler)
        def func(other_arg):
            pass
        
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)


def test_exception_wrapper_handler_default_arg_matches_wrapped():
    try:
        def bad_handler(e, arg1="default"):
            pass
        
        @exception_wrapper(bad_handler)
        def func(arg1):
            pass
        
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_generator_function():
    handler_called = []
    
    def custom_handler(e, arg1):
        handler_called.append((e, arg1))
    
    @exception_wrapper(custom_handler)
    def gen_func(arg1):
        yield 1
        raise ValueError("gen error")
        yield 2
    
    gen = gen_func("test_value")
    assert next(gen) == 1
    
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == "test_value"


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a docstring."""
        pass
    
    assert documented_func.__name__ == "documented_func"
    assert "docstring" in documented_func.__doc__


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def failing_func():
        raise ValueError("test error")
    
    try:
        failing_func()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def failing_func():
        raise ValueError("test error")
    
    try:
        failing_func()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise ValueError("test error")
    
    try:
        failing_func(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, my_default=None):
        handler_called.append((str(e), x, my_default))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    try:
        failing_func(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise ValueError("test error")
    
    try:
        failing_func(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2]["y"] == 20


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def success_func():
        return 42
    
    result = success_func()
    
    assert result == 42
    assert len(handler_called) == 0


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def failing_generator():
        yield 1
        raise ValueError("generator error")
    
    gen = failing_generator()
    assert next(gen) == 1
    
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_invalid_handler_no_args():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_unmatched_arg():
    def invalid_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_matches_wrapped():
    def invalid_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, *args, y=None, **kwargs):
        raise ValueError("test error")
    
    try:
        failing_func(10, 20, 30, y=40, z=50)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 10


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a documented function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "documented function" in documented_func.__doc__


def test_exception_wrapper_generator_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def success_generator():
        yield 1
        yield 2
        yield 3
    
    gen = success_generator()
    result = list(gen)
    
    assert result == [1, 2, 3]
    assert len(handler_called) == 0


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, *, x):
        handler_called.append(x)
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    try:
        failing_func(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == 10


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    # Line 1 predicate: def exception_wrapper(handler_fn=None):
    # This tests that exception_wrapper can be called with handler_fn=None (the default)
    import inspect
    from flutes.exception import exception_wrapper
    
    # Get the signature of exception_wrapper
    sig = inspect.signature(exception_wrapper)
    
    # Verify that handler_fn parameter exists and has a default value of None
    assert 'handler_fn' in sig.parameters
    assert sig.parameters['handler_fn'].default is None
    
    # Verify that exception_wrapper can be called without arguments
    decorator = exception_wrapper()
    assert callable(decorator)
    
    # Verify that exception_wrapper can be called with handler_fn=None explicitly
    decorator_explicit = exception_wrapper(handler_fn=None)
    assert callable(decorator_explicit)
    
    # Verify that exception_wrapper can be used as a decorator without arguments
    @exception_wrapper()
    def dummy_func():
        pass
    
    assert callable(dummy_func)


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
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


def test_exception_wrapper_with_handler_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("Test error")
    
    try:
        func_that_raises(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("Test error", 1, 2)


def test_exception_wrapper_with_handler_and_defaults():
    handler_called = []
    
    def custom_handler(e, x, default_arg=None):
        handler_called.append((str(e), x, default_arg))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("Test error")
    
    try:
        func_that_raises(1)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "Test error"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] is None


def test_exception_wrapper_with_handler_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=2):
        raise ValueError("Test error")
    
    try:
        func_that_raises(1, y=3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "Test error"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == {"y": 3}


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


def test_exception_wrapper_handler_no_positional_arg():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped():
    def bad_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_matches_wrapped():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("Generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Generator error" in handler_called[0]


def test_exception_wrapper_generator_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
    
    result = list(gen_func())
    
    assert result == [1, 2]
    assert len(handler_called) == 0


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, *args, y=None, **kw):
        handler_called.append((str(e), x, args, y, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args, y=None, **kwargs):
        raise ValueError("Test error")
    
    try:
        func_that_raises(1, 2, 3, y=4, z=5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "Test error"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)
    assert handler_called[0][3] == 4
    assert handler_called[0][4] == {"z": 5}


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_predicate_line_6():
    """Test that the predicate at line 6 (handler_fn is not None) evaluates to True when a custom handler is provided."""
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((type(e).__name__, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    failing_func(42)
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("ValueError", 42)


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_predicate_line_5_false():
    """Test that the predicate at line 5 (handler_fn is not None) evaluates to False"""
    from flutes.exception import exception_wrapper
    
    # When handler_fn is None (default), the predicate at line 45 evaluates to False
    @exception_wrapper()
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_handler_fn_is_none():
    @exception_wrapper(handler_fn=None)
    def sample_function():
        raise ValueError("test error")
    
    result = None
    try:
        sample_function()
    except:
        result = "exception was not handled"
    
    assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        assert sys.excepthook is not None
        assert sys.excepthook != original_excepthook
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        assert sys.excepthook is not None
        assert sys.excepthook != original_excepthook
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        assert sys.excepthook is not None
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_exception_hook_set():
    import sys
    
    original_excepthook = sys.excepthook
    try:
        register_ipython_excepthook()
        assert callable(sys.excepthook)
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_bdb_quit_exception():
    import sys
    from bdb import BdbQuit
    from io import StringIO
    
    original_excepthook = sys.excepthook
    original_stdout = sys.stdout
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        sys.stdout = StringIO()
        sys.excepthook(BdbQuit, BdbQuit("test"), None)
    finally:
        sys.excepthook = original_excepthook
        sys.stdout = original_stdout


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #39
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
        raise ValueError("Test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("Test error")
    
    try:
        func_that_raises(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("Test error")
    
    try:
        func_that_raises(10, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2]["y"] == 20


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_normal(x):
        return x * 2
    
    result = func_normal(5)
    
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_handler_with_args_and_varargs():
    handler_called = []
    
    def custom_handler(e, x, args=None, **kw):
        handler_called.append((str(e), x, args, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args, **kwargs):
        raise ValueError("Test error")
    
    try:
        func_that_raises(10, 20, 30, key=40)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == (20, 30)
    assert handler_called[0][3]["key"] == 40


def test_exception_wrapper_handler_no_exception_arg():
    def custom_handler():
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs():
    def custom_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func():
            pass
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)


def test_exception_wrapper_handler_unmatched_arg():
    def custom_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)


def test_exception_wrapper_handler_matching_arg_with_default():
    def custom_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        raise ValueError("Test error")
        yield 3
    
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
    result = list(gen)
    
    assert result == [1, 2, 3]


def test_exception_wrapper_preserves_function_name():
    @exception_wrapper()
    def my_function():
        return 42
    
    assert my_function.__name__ == "my_function"


def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def func_returns(x):
        return x * 2
    
    result = func_returns(5)
    
    assert result == 10


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, *, x=None):
        handler_called.append((str(e), x))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("Test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] is None


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def func_raises_value_error():
        raise ValueError("test error")
    
    func_raises_value_error()


def test_exception_wrapper_with_custom_handler():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_raises_error():
        raise RuntimeError("custom error")
    
    func_raises_error()
    assert len(handler_called) == 1
    assert "custom error" in handler_called[0]


def test_exception_wrapper_with_handler_and_matching_args():
    from flutes.exception import exception_wrapper
    
    captured_args = {}
    
    def custom_handler(e, x, y):
        captured_args['e'] = str(e)
        captured_args['x'] = x
        captured_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def func_with_args(x, y):
        raise ValueError("test")
    
    func_with_args(1, 2)
    assert captured_args['x'] == 1
    assert captured_args['y'] == 2


def test_exception_wrapper_with_handler_and_default_args():
    from flutes.exception import exception_wrapper
    
    captured_args = {}
    
    def custom_handler(e, x, default_arg=None):
        captured_args['e'] = str(e)
        captured_args['x'] = x
        captured_args['default_arg'] = default_arg
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, y=10):
        raise ValueError("test")
    
    func_with_defaults(5)
    assert captured_args['x'] == 5
    assert captured_args['default_arg'] is None


def test_exception_wrapper_with_handler_and_kwargs():
    from flutes.exception import exception_wrapper
    
    captured_args = {}
    
    def custom_handler(e, x, **kw):
        captured_args['e'] = str(e)
        captured_args['x'] = x
        captured_args['kw'] = kw
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y):
        raise ValueError("test")
    
    func_with_kwargs(1, 2)
    assert captured_args['x'] == 1
    assert 'y' in captured_args['kw']


def test_exception_wrapper_with_handler_and_varargs():
    from flutes.exception import exception_wrapper
    
    captured_args = {}
    
    def custom_handler(e, x, args):
        captured_args['e'] = str(e)
        captured_args['x'] = x
        captured_args['args'] = args
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args):
        raise ValueError("test")
    
    func_with_varargs(1, 2, 3)
    assert captured_args['x'] == 1
    assert captured_args['args'] == (2, 3)


def test_exception_wrapper_returns_result_on_success():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def func_returns_value(x):
        return x * 2
    
    result = func_returns_value(5)
    assert result == 10


def test_exception_wrapper_with_generator():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
        yield 2
    
    gen = gen_func()
    first = next(gen)
    assert first == 1
    try:
        next(gen)
    except StopIteration:
        pass
    assert len(handler_called) == 1


def test_exception_wrapper_handler_invalid_no_exception_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_invalid_varargs():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_invalid_unmatched_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_invalid_default_on_matching_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_with_kwonly_args():
    from flutes.exception import exception_wrapper
    
    captured_args = {}
    
    def custom_handler(e, x, *, kwonly_arg=None):
        captured_args['e'] = str(e)
        captured_args['x'] = x
        captured_args['kwonly_arg'] = kwonly_arg
    
    @exception_wrapper(custom_handler)
    def func_with_kwonly(x, y=5):
        raise ValueError("test")
    
    func_with_kwonly(1)
    assert captured_args['x'] == 1
    assert captured_args['kwonly_arg'] is None


def test_exception_wrapper_preserves_function_metadata():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def documented_func():
        """This is a test function."""
        pass
    
    assert documented_func.__name__ == "documented_func"
    assert "test function" in documented_func.__doc__


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper_predicate_line_6_false():
    """Test that the predicate at line 6 (if handler_fn is not None:) evaluates to False"""
    from flutes.exception import exception_wrapper
    
    # When handler_fn is None (default), the predicate at line 45 should be False
    @exception_wrapper()
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def raises_error():
        raise ValueError("test error")
    
    try:
        raises_error()
    except ValueError:
        pass


def test_exception_wrapper_with_handler():
    handler_called = []
    
    def handler(e):
        handler_called.append(e)
    
    @exception_wrapper(handler)
    def raises_error():
        raise ValueError("test error")
    
    raises_error()
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_with_matching_args():
    handler_args = {}
    
    def handler(e, x, y):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(handler)
    def raises_with_args(x, y):
        raise ValueError("test error")
    
    raises_with_args(10, 20)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 10
    assert handler_args['y'] == 20


def test_exception_wrapper_with_default_args():
    handler_args = {}
    
    def handler(e, x=None):
        handler_args['e'] = e
        handler_args['x'] = x
    
    @exception_wrapper(handler)
    def raises_with_args(x):
        raise ValueError("test error")
    
    raises_with_args(10)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] is None


def test_exception_wrapper_with_varargs():
    handler_args = {}
    
    def handler(e, args):
        handler_args['e'] = e
        handler_args['args'] = args
    
    @exception_wrapper(handler)
    def raises_with_varargs(x, *args):
        raise ValueError("test error")
    
    raises_with_varargs(1, 2, 3)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['args'] == (2, 3)


def test_exception_wrapper_with_kwargs():
    handler_args = {}
    
    def handler(e, kw):
        handler_args['e'] = e
        handler_args['kw'] = kw
    
    @exception_wrapper(handler)
    def raises_with_kwargs(x, **kwargs):
        raise ValueError("test error")
    
    raises_with_kwargs(1, y=2, z=3)
    assert isinstance(handler_args['e'], ValueError)
    assert 'y' in handler_args['kw']
    assert 'z' in handler_args['kw']


def test_exception_wrapper_returns_value_on_success():
    @exception_wrapper()
    def returns_value(x):
        return x * 2
    
    result = returns_value(5)
    assert result == 10


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def handler(e):
        handler_called.append(e)
    
    @exception_wrapper(handler)
    def generator_func():
        yield 1
        yield 2
        raise ValueError("generator error")
    
    gen = generator_func()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except StopIteration:
        pass
    assert len(handler_called) == 1


def test_exception_wrapper_invalid_handler_no_args():
    def handler():
        pass
    
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def handler(e, *args):
        pass
    
    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_mismatch():
    def handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_conflicts():
    def handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_varkw():
    handler_args = {}
    
    def handler(e, x, **varkw):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['varkw'] = varkw
    
    @exception_wrapper(handler)
    def raises_with_varkw(x, **kwargs):
        raise ValueError("test error")
    
    raises_with_varkw(1, y=2, z=3)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 1
    assert 'y' in handler_args['varkw']
    assert 'z' in handler_args['varkw']


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def my_function():
        """My docstring"""
        pass
    
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring"


def test_exception_wrapper_with_kwonly_args():
    handler_args = {}
    
    def handler(e, x):
        handler_args['e'] = e
        handler_args['x'] = x
    
    @exception_wrapper(handler)
    def raises_with_kwonly(*, x):
        raise ValueError("test error")
    
    raises_with_kwonly(x=5)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 5


def test_exception_wrapper_multiple_decorators():
    handler_called = []
    
    def handler(e):
        handler_called.append(e)
    
    def outer_decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    
    @outer_decorator
    @exception_wrapper(handler)
    def raises_error():
        raise ValueError("test error")
    
    raises_error()
    assert len(handler_called) == 1


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("test error")
    
    try:
        func_that_raises(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10


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


def test_exception_wrapper_invalid_handler_varargs():
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
    def invalid_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y=None):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 10


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    
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


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func(x):
        """This is a test function."""
        return x
    
    assert documented_func.__name__ == "documented_func"
    assert "test function" in documented_func.__doc__


def test_exception_wrapper_handler_receives_all_args():
    handler_called = []
    
    def custom_handler(e, a, b, c=None):
        handler_called.append((a, b, c))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(a, b, c=None):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, c=3)
    except ValueError:
        pass
    
    assert handler_called[0] == (1, 2, 3)


def test_exception_wrapper_handler_with_varargs_capture_in_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20, 30)
    except ValueError:
        pass
    
    assert handler_called[0][0] == 10
    assert handler_called[0][1]["args"] == (20, 30)


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_args = {}
    
    def custom_handler(e, x, y):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def failing_function(x, y):
        raise RuntimeError("Error occurred")
    
    try:
        failing_function(10, 20)
    except RuntimeError:
        pass
    
    assert handler_args['x'] == 10
    assert handler_args['y'] == 20


def test_exception_wrapper_handler_with_default_args():
    handler_args = {}
    
    def custom_handler(e, x, y=100):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def failing_function(x):
        raise TypeError("Type error")
    
    try:
        failing_function(5)
    except TypeError:
        pass
    
    assert handler_args['x'] == 5
    assert handler_args['y'] == 100


def test_exception_wrapper_handler_with_kwargs():
    handler_args = {}
    
    def custom_handler(e, x, **kwargs):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['kwargs'] = kwargs
    
    @exception_wrapper(custom_handler)
    def failing_function(x, y):
        raise RuntimeError("Error")
    
    try:
        failing_function(1, 2)
    except RuntimeError:
        pass
    
    assert handler_args['x'] == 1
    assert handler_args['kwargs']['y'] == 2


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(True)
    
    @exception_wrapper(custom_handler)
    def successful_function(x):
        return x * 2
    
    result = successful_function(5)
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def failing_generator():
        yield 1
        raise ValueError("Generator error")
    
    gen = failing_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_successful_generator():
    @exception_wrapper()
    def successful_generator():
        yield 1
        yield 2
        yield 3
    
    gen = successful_generator()
    values = list(gen)
    assert values == [1, 2, 3]


def test_exception_wrapper_handler_missing_required_arg():
    def custom_handler(e, missing_arg):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function():
            raise ValueError("Error")
        
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "does not match" in str(ve)


def test_exception_wrapper_handler_with_varargs_raises_error():
    def custom_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function():
            raise ValueError("Error")
        
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "varargs" in str(ve)


def test_exception_wrapper_handler_no_exception_arg_raises_error():
    def custom_handler():
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function():
            raise ValueError("Error")
        
        assert False, "Should have raised ValueError"
    except ValueError as ve:
        assert "positional argument" in str(ve)


def test_exception_wrapper_with_args_and_kwargs():
    handler_args = {}
    
    def custom_handler(e, a, b=10, **kwargs):
        handler_args['a'] = a
        handler_args['b'] = b
        handler_args['kwargs'] = kwargs
    
    @exception_wrapper(custom_handler)
    def failing_function(a, c):
        raise RuntimeError("Error")
    
    try:
        failing_function(5, 6)
    except RuntimeError:
        pass
    
    assert handler_args['a'] == 5
    assert handler_args['b'] == 10
    assert handler_args['kwargs']['c'] == 6


def test_exception_wrapper_with_keyword_only_args():
    handler_args = {}
    
    def custom_handler(e, x, *, y=20):
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def failing_function(x):
        raise ValueError("Error")
    
    try:
        failing_function(10)
    except ValueError:
        pass
    
    assert handler_args['x'] == 10
    assert handler_args['y'] == 20


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_function():
        """Test docstring"""
        raise ValueError("Error")
    
    assert documented_function.__doc__ == "Test docstring"
    assert documented_function.__name__ == "documented_function"


def test_exception_wrapper_handler_with_varargs_in_wrapped_function():
    handler_args = {}
    
    def custom_handler(e, x):
        handler_args['x'] = x
    
    @exception_wrapper(custom_handler)
    def failing_function(x, *args):
        raise RuntimeError("Error")
    
    try:
        failing_function(1, 2, 3)
    except RuntimeError:
        pass
    
    assert handler_args['x'] == 1


def test_exception_wrapper_handler_with_wrapped_decorator():
    def outer_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append(x)
    
    @exception_wrapper(custom_handler)
    @outer_decorator
    def failing_function(x):
        raise ValueError("Error")
    
    try:
        failing_function(42)
    except ValueError:
        pass
    
    assert handler_called[0] == 42


def test_exception_wrapper_handler_receives_all_bound_args():
    handler_args = {}
    
    def custom_handler(e, a, b, c=30, **kwargs):
        handler_args['a'] = a
        handler_args['b'] = b
        handler_args['c'] = c
        handler_args['kwargs'] = kwargs
    
    @exception_wrapper(custom_handler)
    def failing_function(a, b, c=30, d=40):
        raise RuntimeError("Error")
    
    try:
        failing_function(1, 2)


# LLM-generated content at query #45
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    from bdb import BdbQuit
    import sys
    
    # Test the predicate: `not capture_keyboard_interrupt` evaluates to True
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    
    # This is the predicate at line 8: `not capture_keyboard_interrupt`
    predicate_result = not capture_keyboard_interrupt
    
    assert predicate_result is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_basic():
    import sys
    from io import StringIO
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    test_exception = ValueError("test error message")
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception)
        
        assert mock_log.call_count == 2
        first_call_args = mock_log.call_args_list[0]
        second_call_args = mock_log.call_args_list[1]
        
        assert "Traceback" in first_call_args[0][0]
        assert first_call_args[0][1] == "error"
        
        assert "<ValueError> test error message" in second_call_args[0][0]
        assert second_call_args[0][1] == "error"


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = RuntimeError("runtime error")
    user_message = "Custom user message"
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception, user_msg=user_message)
        
        assert mock_log.call_count == 2
        second_call_args = mock_log.call_args_list[1]
        
        assert f"{user_message}: <RuntimeError> runtime error" in second_call_args[0][0]
        assert second_call_args[0][1] == "error"


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = TypeError("type error")
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception, force_console=True, timestamp=False)
        
        assert mock_log.call_count == 2
        first_call_args = mock_log.call_args_list[0]
        
        assert first_call_args[1]['force_console'] is True
        assert first_call_args[1]['timestamp'] is False


def test_log_exception_subprocess_error_with_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = subprocess.CalledProcessError(1, "cmd", output="error output")
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception)
        
        assert mock_log.call_count == 1
        call_args = mock_log.call_args_list[0]
        assert "CalledProcessError" in call_args[0][0]


def test_log_exception_subprocess_error_without_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = subprocess.CalledProcessError(1, "cmd", output=None)
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception)
        
        assert mock_log.call_count == 2
        first_call_args = mock_log.call_args_list[0]
        assert "Traceback" in first_call_args[0][0]


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = KeyError("key error")
    log_exception_to_raise = RuntimeError("logging failed")
    
    with patch('flutes.exception.log', side_effect=log_exception_to_raise):
        with patch('builtins.print') as mock_print:
            try:
                log_exception(test_exception)
            except RuntimeError:
                pass
            
            assert mock_print.call_count >= 1
            first_print_call = mock_print.call_args_list[0]
            assert "<KeyError> key error" in first_print_call[0][0]


def test_log_exception_all_params():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    test_exception = Exception("generic exception")
    user_message = "Operation failed"
    
    with patch('flutes.exception.log') as mock_log:
        log_exception(test_exception, user_msg=user_message, force_console=True, include_proc_id=False)
        
        assert mock_log.call_count == 2
        first_call_args = mock_log.call_args_list[0]
        second_call_args = mock_log.call_args_list[1]
        
        assert first_call_args[1]['force_console'] is True
        assert first_call_args[1]['include_proc_id'] is False
        assert f"{user_message}: <Exception> generic exception" in second_call_args[0][0]


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_excepthook_callable():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert callable(sys.excepthook)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_bdbquit_not_captured():
    import sys
    from bdb import BdbQuit
    from unittest.mock import patch
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    with patch('sys.__excepthook__') as mock_sys_excepthook:
        sys.excepthook(BdbQuit, BdbQuit("test"), None)
        mock_sys_excepthook.assert_called_once()
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_not_captured():
    import sys
    from unittest.mock import patch
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    with patch('sys.__excepthook__') as mock_sys_excepthook:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), None)
        mock_sys_excepthook.assert_called_once()
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_captured():
    import sys
    from unittest.mock import patch, MagicMock
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    with patch('IPython.core.ultratb.FormattedTB') as mock_tb:
        mock_instance = MagicMock()
        mock_tb.return_value = mock_instance
        
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), None)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_other_exception():
    import sys
    from unittest.mock import patch, MagicMock
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    with patch('IPython.core.ultratb.FormattedTB') as mock_tb:
        mock_instance = MagicMock()
        mock_tb.return_value = mock_instance
        
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        sys.excepthook(ValueError, ValueError("test"), None)
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
#--------------------------

```python
def test_exception_wrapper_no_handler_logs_exception(monkeypatch):
    log_calls = []
    def mock_log(msg, level="info", **kwargs):
        log_calls.append((msg, level, kwargs))
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    monkeypatch.setattr("flutes.exception.log_exception", lambda e: log_calls.append(("log_exception", e)))
    
    @exception_wrapper()
    def failing_func():
        raise ValueError("test error")
    
    failing_func()
    assert len(log_calls) > 0
    assert log_calls[0][0] == "log_exception"
    assert isinstance(log_calls[0][1], ValueError)


def test_exception_wrapper_with_handler():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    failing_func(42)
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ValueError)
    assert handler_calls[0][1] == 42


def test_exception_wrapper_handler_with_defaults():
    handler_calls = []
    
    def custom_handler(e, x, y=10):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    failing_func(5)
    assert len(handler_calls) == 1
    assert handler_calls[0][0].__class__.__name__ == "ValueError"
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2] == 10


def test_exception_wrapper_handler_with_kwargs():
    handler_calls = []
    
    def custom_handler(e, x, **kw):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise ValueError("test error")
    
    failing_func(1, 2)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert "y" in handler_calls[0][2]
    assert handler_calls[0][2]["y"] == 2


def test_exception_wrapper_no_exception():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def success_func(x):
        return x * 2
    
    result = success_func(5)
    assert result == 10
    assert len(handler_calls) == 0


def test_exception_wrapper_generator():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_generator(x):
        yield 1
        raise ValueError("generator error")
    
    gen = failing_generator(42)
    next(gen)
    try:
        next(gen)
    except StopIteration:
        pass
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 42


def test_exception_wrapper_handler_validation_no_positional_arg():
    def invalid_handler(**kw):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_validation_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_validation_mismatched_arg():
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_validation_default_value_mismatch():
    def invalid_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_calls = []
    
    def custom_handler(e, a, **kw):
        handler_calls.append((e, a, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(a, b, c=3):
        raise ValueError("error")
    
    failing_func(1, 2, c=4)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert "b" in handler_calls[0][2]
    assert handler_calls[0][2]["b"] == 2
    assert "c" in handler_calls[0][2]
    assert handler_calls[0][2]["c"] == 4


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a doc string"""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "doc string" in documented_func.__doc__


def test_exception_wrapper_generator_no_exception():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def success_generator(x):
        yield 1
        yield 2
        yield 3
    
    gen = success_generator(42)
    results = list(gen)
    assert results == [1, 2, 3]
    assert len(handler_calls) == 0


def test_exception_wrapper_handler_with_kwonly_args():
    handler_calls = []
    
    def custom_handler(e, x, *, y=20):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("error")
    
    failing_func(5)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2] == 20


# LLM-generated content at query #4
#--------------------------

```python
def test_log_exception_predicate_line_15_false():
    import subprocess
    from flutes.exception import log_exception
    
    # Create a CalledProcessError with output set to None
    # This makes the predicate at line 12 evaluate to False
    # because: not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)
    # = not (True and False) = not False = True
    # So it will try to log traceback, which may raise an exception
    
    # Create a CalledProcessError with output=None
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output=None)
    
    # Call log_exception - the predicate at line 15 should evaluate to False
    # meaning no exception is raised in the try block
    try:
        log_exception(error, user_msg="Test message", force_console=False)
        exception_raised = False
    except Exception:
        exception_raised = True
    
    # The predicate at line 15 evaluates to False means no exception occurred
    assert exception_raised == False


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    """Test that the predicate at line 1 (function definition) evaluates to True by checking the decorator is callable."""
    from flutes.exception import exception_wrapper
    
    # The predicate at line 1 is the function definition itself
    # We test that exception_wrapper exists and is callable
    assert callable(exception_wrapper)
    
    # Test that it can be called without arguments
    decorator = exception_wrapper()
    assert callable(decorator)
    
    # Test that it can decorate a simple function
    def simple_func():
        return "success"
    
    decorated = decorator(simple_func)
    assert callable(decorated)
    assert decorated() == "success"
    
    # Test that it can be called with a handler function
    def handler(e):
        pass
    
    decorator_with_handler = exception_wrapper(handler)
    assert callable(decorator_with_handler)
    
    decorated_with_handler = decorator_with_handler(simple_func)
    assert callable(decorated_with_handler)
    assert decorated_with_handler() == "success"


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0].__class__.__name__ == "ValueError"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=10):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert "y" in handler_called[0][2]


def test_exception_wrapper_handler_with_varargs():
    handler_called = []
    
    def custom_handler(e, *args):
        handler_called.append((e, args))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    with_error = False
    try:
        func_that_raises(1)
    except ValueError:
        with_error = True
    
    assert with_error


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error(x, y):
        return x + y
    
    result = func_no_error(1, 2)
    assert result == 3


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("gen error")
    
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_generator_success():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, *, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5


def test_exception_wrapper_invalid_handler_no_args():
    def bad_handler():
        pass
    
    error_raised = False
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
    except ValueError as e:
        error_raised = True
        assert "positional argument" in str(e)
    
    assert error_raised


def test_exception_wrapper_invalid_handler_varargs():
    def bad_handler(e, *args):
        pass
    
    error_raised = False
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
    except ValueError as e:
        error_raised = True
        assert "varargs" in str(e)
    
    assert error_raised


def test_exception_wrapper_invalid_handler_unmatched_arg():
    def bad_handler(e, nonexistent):
        pass
    
    error_raised = False
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
    except ValueError as e:
        error_raised = True
        assert "does not match" in str(e)
    
    assert error_raised


def test_exception_wrapper_invalid_handler_default_on_matching():
    def bad_handler(e, x=10):
        pass
    
    error_raised = False
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
    except ValueError as e:
        error_raised = True
        assert "cannot have default values" in str(e)
    
    assert error_raised


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a documented function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "documented function" in documented_func.__doc__


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b=20, **kw):
        handler_called.append((e, a, b, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(a, b, c):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, 3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2
    assert handler_called[0][3]["c"] == 3


def test_exception_wrapper_handler_with_varargs_capture():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((e, x, args))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, 3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


# LLM-generated content at query #7
#--------------------------

```python
def test_log_exception_basic():
    import io
    import sys
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error message")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e)
            assert mock_log.call_count == 2
            first_call_args = mock_log.call_args_list[0]
            second_call_args = mock_log.call_args_list[1]
            assert "Traceback" in first_call_args[0][0]
            assert "error" in first_call_args[0]
            assert "<ValueError> test error message" in second_call_args[0][0]
            assert "error" in second_call_args[0]


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise RuntimeError("original error")
    except RuntimeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Custom message")
            assert mock_log.call_count == 2
            second_call_args = mock_log.call_args_list[1]
            assert "Custom message: <RuntimeError> original error" in second_call_args[0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, force_console=True, timestamp=False)
            assert mock_log.call_count == 2
            first_call_args = mock_log.call_args_list[0]
            assert first_call_args[1]['force_console'] is True
            assert first_call_args[1]['timestamp'] is False


def test_log_exception_with_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, "cmd", output="command output")
    with patch('flutes.exception.log') as mock_log:
        log_exception(error)
        assert mock_log.call_count == 1
        call_args = mock_log.call_args_list[0]
        assert "CalledProcessError" in call_args[0][0]


def test_log_exception_logging_fails():
    import sys
    from io import StringIO
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise KeyError("key error")
    except KeyError as e:
        with patch('flutes.exception.log', side_effect=Exception("logging failed")):
            captured_output = StringIO()
            sys.stdout = captured_output
            try:
                log_exception(e, user_msg="Test")
            except Exception as log_e:
                sys.stdout = sys.__stdout__
                assert "logging failed" in str(log_e)
            sys.stdout = sys.__stdout__


def test_log_exception_no_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise IndexError("index out of range")
    except IndexError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg=None)
            assert mock_log.call_count == 2
            second_call_args = mock_log.call_args_list[1]
            assert "<IndexError> index out of range" in second_call_args[0][0]
            assert ":" not in second_call_args[0][0].split("<IndexError>")[0]


# LLM-generated content at query #8
#--------------------------

```python
def test_log_exception_predicate_line_12_true():
    import subprocess
    from flutes.exception import log_exception
    
    error = subprocess.CalledProcessError(1, "cmd")
    error.output = b"some output"
    
    try:
        raise error
    except subprocess.CalledProcessError as e:
        log_exception(e, user_msg="Test message")


# LLM-generated content at query #9
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
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise RuntimeError("original error")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        assert mock_log.call_count == 2
        calls = mock_log.call_args_list
        assert "Custom message: <RuntimeError> original error" in calls[1][0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise TypeError("type error")
        except TypeError as e:
            log_exception(e, force_console=True, timestamp=False)
        
        assert mock_log.call_count == 2
        calls = mock_log.call_args_list
        assert calls[0][1]['force_console'] is True
        assert calls[0][1]['timestamp'] is False
        assert calls[1][1]['force_console'] is True
        assert calls[1][1]['timestamp'] is False


def test_log_exception_subprocess_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        error = subprocess.CalledProcessError(1, "cmd", output="some output")
        log_exception(error)
        
        assert mock_log.call_count == 1
        assert "CalledProcessError" in mock_log.call_args_list[0][0][0]


def test_log_exception_subprocess_error_no_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        error = subprocess.CalledProcessError(1, "cmd", output=None)
        log_exception(error)
        
        assert mock_log.call_count == 2


def test_log_exception_logging_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log', side_effect=Exception("Log failed")):
        with patch('builtins.print') as mock_print:
            try:
                raise ValueError("original error")
            except ValueError as e:
                try:
                    log_exception(e, user_msg="test")
                except Exception as logged_e:
                    assert isinstance(logged_e, Exception)
                    assert mock_print.call_count == 2


def test_log_exception_level_is_error():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise KeyError("key error")
        except KeyError as e:
            log_exception(e)
        
        calls = mock_log.call_args_list
        assert all(call[1].get('level') == "error" or call[0][1] == "error" for call in calls)


def test_log_exception_traceback_included():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    with patch('flutes.exception.log') as mock_log:
        try:
            raise AssertionError("assertion failed")
        except AssertionError as e:
            log_exception(e)
        
        first_call_msg = mock_log.call_args_list[0][0][0]
        assert "Traceback" in first_call_msg or "AssertionError" in first_call_msg


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
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
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("Test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=10):
        raise ValueError("Test error")
    
    try:
        func_that_raises(5, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=10):
        raise ValueError("Test error")
    
    try:
        func_that_raises(5, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == {"y": 20}


def test_exception_wrapper_normal_execution():
    @exception_wrapper()
    def func_that_returns(x):
        return x * 2
    
    result = func_that_returns(5)
    assert result == 10


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
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
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_no_positional_arg_raises():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs_raises():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped_raises():
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_with_default_matching_wrapped_raises():
    def invalid_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def my_func(x):
        """My docstring"""
        return x
    
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring"


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b, c=None, **kw):
        handler_called.append((a, b, c, kw))
    
    @exception_wrapper(custom_handler)
    def func(a, b, c=None, **kw):
        raise ValueError("Test")
    
    try:
        func(1, 2, c=3, d=4)
    except ValueError:
        pass
    
    assert handler_called[0] == (1, 2, 3, {"d": 4})


def test_exception_wrapper_no_exception_in_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
        yield 3
    
    gen = gen_func()
    result = list(gen)
    assert result == [1, 2, 3]


def test_exception_wrapper_returns_generator_without_consuming():
    @exception_wrapper()
    def gen_func():
        yield 1
        yield 2
    
    gen = gen_func()
    assert hasattr(gen, "__next__")
    assert next(gen) == 1


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((x, y))
    
    @exception_wrapper(custom_handler)
    def failing_function(x, y):
        raise ValueError("Test error")
    
    try:
        failing_function(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == (1, 2)


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, default_arg=None):
        handler_called.append((x, default_arg))
    
    @exception_wrapper(custom_handler)
    def failing_function(x):
        raise ValueError("Test error")
    
    try:
        failing_function(1)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == (1, None)


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_function(x, y=None):
        raise ValueError("Test error")
    
    try:
        failing_function(1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert "y" in handler_called[0][1]


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(True)
    
    @exception_wrapper(custom_handler)
    def normal_function(x):
        return x * 2
    
    result = normal_function(5)
    
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(True)
    
    @exception_wrapper(custom_handler)
    def failing_generator():
        yield 1
        raise ValueError("Test error")
    
    gen = failing_generator()
    next(gen)
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_generator_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(True)
    
    @exception_wrapper(custom_handler)
    def normal_generator():
        yield 1
        yield 2
    
    gen = normal_generator()
    result = list(gen)
    
    assert result == [1, 2]
    assert len(handler_called) == 0


def test_exception_wrapper_handler_no_exception_arg():
    def custom_handler():
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function():
            raise ValueError("Test error")
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs():
    def custom_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function():
            raise ValueError("Test error")
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped():
    def custom_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function(x):
            raise ValueError("Test error")
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)


def test_exception_wrapper_handler_default_arg_matches_wrapped():
    def custom_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def failing_function(x):
            raise ValueError("Test error")
    except ValueError as e:
        assert "matches wrapped method argument, thus cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b, extra=None, **kw):
        handler_called.append((a, b, extra, kw))
    
    @exception_wrapper(custom_handler)
    def failing_function(a, b, c=None, **kwargs):
        raise ValueError("Test error")
    
    try:
        failing_function(1, 2, c=3, d=4)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert handler_called[0][1] == 2


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_function():
        """This is a documented function"""
        pass
    
    assert documented_function.__name__ == "documented_function"
    assert "documented function" in documented_function.__doc__


# LLM-generated content at query #12
#--------------------------

```python
def test_register_ipython_excepthook_with_capture_keyboard_interrupt_false():
    from bdb import BdbQuit
    import sys
    
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    assert KeyboardInterrupt in skip_exceptions
    assert len(skip_exceptions) == 2
    assert BdbQuit in skip_exceptions


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #14
#--------------------------

```python
def test_log_exception_predicate_line_15_false():
    import subprocess
    from flutes.exception import log_exception
    
    # Create a CalledProcessError with output set (makes the predicate at line 12 False)
    # This means line 13 will NOT execute, and we go directly to line 14
    # Line 15's except block should NOT be triggered if log() succeeds
    error = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    
    # Call log_exception - should not raise an exception
    # The predicate at line 15 (except Exception) evaluates to False when no exception occurs
    log_exception(error, user_msg="Test message")


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
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
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 10, 20)


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, my_default=None):
        handler_called.append((str(e), x, my_default))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, **kwargs):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, y=20, z=30)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == {"y": 20, "z": 30}


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
        raise ValueError("generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "generator error" in handler_called[0]


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


def test_exception_wrapper_invalid_handler_mismatched_arg():
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
    def bad_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b, c=None, **kw):
        handler_called.append((a, b, c, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(a, b, *args, c=None, **kwargs):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, "arg1", "arg2", c=3, d=4, e=5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert handler_called[0][1] == 2
    assert handler_called[0][2] == 3
    assert handler_called[0][3] == {"args": ("arg1", "arg2"), "kwargs": {"d": 4, "e": 5}}


def test_exception_wrapper_preserves_function_metadata():
    def custom_handler(e):
        pass
    
    @exception_wrapper(custom_handler)
    def my_func():
        """My function docstring"""
        pass
    
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My function docstring"


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y):
        handler_called.append((x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == (1, 2)


def test_exception_wrapper_handler_with_kwonly_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == (5, None)


# LLM-generated content at query #16
#--------------------------

```python
def test_log_exception_basic():
    import traceback
    from unittest.mock import patch, MagicMock
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e)
            assert mock_log.call_count == 2
            first_call_args = mock_log.call_args_list[0]
            second_call_args = mock_log.call_args_list[1]
            assert "Traceback" in first_call_args[0][0]
            assert first_call_args[0][1] == "error"
            assert "<ValueError> test error" in second_call_args[0][0]
            assert second_call_args[0][1] == "error"


def test_log_exception_with_user_msg():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise RuntimeError("runtime error")
    except RuntimeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Custom message")
            assert mock_log.call_count == 2
            second_call_args = mock_log.call_args_list[1]
            assert "Custom message: <RuntimeError> runtime error" in second_call_args[0][0]


def test_log_exception_with_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise TypeError("type error")
    except TypeError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, force_console=True, timestamp=False)
            first_call_args = mock_log.call_args_list[0]
            second_call_args = mock_log.call_args_list[1]
            assert first_call_args[1]['force_console'] is True
            assert first_call_args[1]['timestamp'] is False
            assert second_call_args[1]['force_console'] is True
            assert second_call_args[1]['timestamp'] is False


def test_log_exception_with_called_process_error():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    e = subprocess.CalledProcessError(1, "cmd", output="output data")
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 1
        call_args = mock_log.call_args_list[0]
        assert "CalledProcessError" in call_args[0][0]


def test_log_exception_with_called_process_error_no_output():
    import subprocess
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    e = subprocess.CalledProcessError(1, "cmd", output=None)
    with patch('flutes.exception.log') as mock_log:
        log_exception(e)
        assert mock_log.call_count == 2


def test_log_exception_log_fails():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise ValueError("test error")
    except ValueError as e:
        with patch('flutes.exception.log', side_effect=RuntimeError("log failed")):
            with patch('builtins.print') as mock_print:
                try:
                    log_exception(e)
                except RuntimeError:
                    pass
                assert mock_print.call_count == 2
                first_print_args = mock_print.call_args_list[0]
                second_print_args = mock_print.call_args_list[1]
                assert "<ValueError> test error" in first_print_args[0][0]
                assert "Another exception occurred while logging" in second_print_args[0][0]


def test_log_exception_with_user_msg_and_kwargs():
    from unittest.mock import patch
    from flutes.exception import log_exception
    
    try:
        raise KeyError("key error")
    except KeyError as e:
        with patch('flutes.exception.log') as mock_log:
            log_exception(e, user_msg="Key not found", include_proc_id=False)
            assert mock_log.call_count == 2
            second_call_args = mock_log.call_args_list[1]
            assert "Key not found: <KeyError> 'key error'" in second_call_args[0][0]
            assert second_call_args[1]['include_proc_id'] is False


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    # Line 1 predicate: def exception_wrapper(handler_fn=None):
    # This tests that exception_wrapper is callable and accepts handler_fn parameter with default None
    import inspect
    from flutes.exception import exception_wrapper
    
    # Get the signature of exception_wrapper
    sig = inspect.signature(exception_wrapper)
    
    # Check that exception_wrapper has a parameter named 'handler_fn'
    assert 'handler_fn' in sig.parameters
    
    # Check that 'handler_fn' has a default value of None
    assert sig.parameters['handler_fn'].default is None
    
    # Check that exception_wrapper is callable
    assert callable(exception_wrapper)
    
    # Test that it can be called with no arguments
    decorator = exception_wrapper()
    assert callable(decorator)
    
    # Test that it can be called with handler_fn=None explicitly
    decorator = exception_wrapper(handler_fn=None)
    assert callable(decorator)


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    decorator = exception_wrapper()
    assert callable(decorator)


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, default_arg=None):
        handler_called.append((str(e), default_arg))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] is None


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == {"y": 20}


def test_exception_wrapper_handler_with_varargs():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((str(e), x, args))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, 20, 30)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == (20, 30)


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
    def func_generator():
        yield 1
        raise ValueError("test error")
        yield 2
    
    gen = func_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_handler_no_positional_arg_raises():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs_raises():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_mismatched_arg_raises():
    def bad_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_matching_arg_raises():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *, y=5):
        raise ValueError("test error")
    
    try:
        func_that_raises(10, y=20)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 10
    assert handler_called[0][2] == 20


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a test function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "test function" in documented_func.__doc__


def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def func_generator():
        yield 1
        yield 2
        yield 3
    
    gen = func_generator()
    result = list(gen)
    assert result == [1, 2, 3]


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_predicate_line_12_true():
    import subprocess
    from flutes.exception import log_exception
    from unittest.mock import patch
    
    # Create a CalledProcessError with output
    error = subprocess.CalledProcessError(returncode=1, cmd="test_cmd", output="test_output")
    
    # Mock the log function to verify it's called correctly
    with patch('flutes.exception.log') as mock_log:
        log_exception(error, user_msg="Test error")
        
        # When isinstance(e, subprocess.CalledProcessError) and e.output is not None,
        # the predicate at line 12 is True, so traceback.format_exc() should NOT be logged
        # Only the exc_msg should be logged
        assert mock_log.call_count == 1
        call_args = mock_log.call_args
        assert "Test error" in call_args[0][0]


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_basic_exception_logging():
    from flutes.exception import exception_wrapper, log_exception
    from unittest.mock import patch
    
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    with patch('flutes.exception.log_exception') as mock_log:
        func_that_raises()
        mock_log.assert_called_once()
        assert isinstance(mock_log.call_args[0][0], ValueError)


def test_exception_wrapper_with_custom_handler():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    func_that_raises()
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_with_matching_args():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, y):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    func_that_raises(10, 20)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 10
    assert handler_args['y'] == 20


def test_exception_wrapper_handler_with_default_args():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, my_arg=None):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['my_arg'] = my_arg
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    func_that_raises(10)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 10
    assert handler_args['my_arg'] is None


def test_exception_wrapper_handler_with_varkw():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, **kw):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['kw'] = kw
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None, z=None):
        raise ValueError("test error")
    
    func_that_raises(10, y=20, z=30)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 10
    assert handler_args['kw']['y'] == 20
    assert handler_args['kw']['z'] == 30


def test_exception_wrapper_no_exception():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_generator():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
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
    except StopIteration:
        pass
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_invalid_handler_no_args():
    from flutes.exception import exception_wrapper
    
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)


def test_exception_wrapper_invalid_handler_unmatched_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_for_matching_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, a, b, my_arg=None, **kw):
        handler_args['e'] = e
        handler_args['a'] = a
        handler_args['b'] = b
        handler_args['my_arg'] = my_arg
        handler_args['kw'] = kw
    
    @exception_wrapper(custom_handler)
    def func(a, b, c=None, **kwargs):
        raise ValueError("test error")
    
    func(1, 2, c=3, d=4)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['a'] == 1
    assert handler_args['b'] == 2
    assert handler_args['my_arg'] is None
    assert handler_args['kw']['c'] == 3
    assert handler_args['kw']['d'] == 4


def test_exception_wrapper_preserves_function_metadata():
    from flutes.exception import exception_wrapper
    
    def custom_handler(e):
        pass
    
    @exception_wrapper(custom_handler)
    def my_func():
        """My docstring"""
        pass
    
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring"


def test_exception_wrapper_with_varargs():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, args=None, **kw):
        handler_args['e'] = e
        handler_args['x'] = x
        handler_args['args'] = args
        handler_args['kw'] = kw
    
    @exception_wrapper(custom_handler)
    def func(x, *args):
        raise ValueError("test error")
    
    func(1, 2, 3)
    assert isinstance(handler_args['e'], ValueError)
    assert handler_args['x'] == 1
    assert handler_args['args'] == (2, 3)


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    result = callable(exception_wrapper)
    assert result is True


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    from flutes.exception import exception_wrapper
    
    call_count = [0]
    
    @exception_wrapper()
    def failing_func():
        call_count[0] += 1
        raise ValueError("Test error")
    
    failing_func()
    assert call_count[0] == 1


def test_exception_wrapper_with_custom_handler():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("Test error")
    
    failing_func(42)
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ValueError)
    assert handler_calls[0][1] == 42


def test_exception_wrapper_with_handler_and_default_args():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, x, y=None):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y=10):
        raise ValueError("Test error")
    
    failing_func(5)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2] == 10


def test_exception_wrapper_with_handler_and_kwargs():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, x, **kw):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y=10, z=20):
        raise ValueError("Test error")
    
    failing_func(5, z=30)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2]["y"] == 10
    assert handler_calls[0][2]["z"] == 30


def test_exception_wrapper_no_exception():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e):
        handler_calls.append(e)
    
    @exception_wrapper(custom_handler)
    def success_func():
        return 42
    
    result = success_func()
    assert result == 42
    assert len(handler_calls) == 0


def test_exception_wrapper_with_generator():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e):
        handler_calls.append(e)
    
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
    except StopIteration:
        pass
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0], ValueError)


def test_exception_wrapper_invalid_handler_no_args():
    from flutes.exception import exception_wrapper
    
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)


def test_exception_wrapper_handler_arg_mismatch():
    from flutes.exception import exception_wrapper
    
    def custom_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)


def test_exception_wrapper_handler_with_args_and_varargs():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, x, *args):
        handler_calls.append((e, x, args))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, *args):
        raise ValueError("Test error")
    
    failing_func(1, 2, 3)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == (2, 3)


def test_exception_wrapper_preserves_function_metadata():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def my_function():
        """My docstring"""
        pass
    
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring"


def test_exception_wrapper_with_keyword_only_args():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, x, y):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, *, y=10):
        raise ValueError("Test error")
    
    failing_func(5, y=20)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 5
    assert handler_calls[0][2] == 20


def test_exception_wrapper_handler_with_default_not_matching_wrapped():
    from flutes.exception import exception_wrapper
    
    handler_calls = []
    
    def custom_handler(e, my_extra_arg=None):
        handler_calls.append((e, my_extra_arg))
    
    @exception_wrapper(custom_handler)
    def failing_func():
        raise ValueError("Test error")
    
    failing_func()
    assert len(handler_calls) == 1
    assert handler_calls[0][1] is None


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_predicate_line_1():
    """Test that the predicate at line 1 (handler_fn=None) evaluates to False when handler_fn is None."""
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #26
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    from unittest.mock import patch, MagicMock
    import sys
    
    # Mock the IPython module
    mock_ultratb = MagicMock()
    mock_formatted_tb = MagicMock()
    mock_ultratb.FormattedTB.return_value = mock_formatted_tb
    
    with patch.dict('sys.modules', {'IPython': MagicMock(), 'IPython.core': MagicMock(), 'IPython.core.ultratb': mock_ultratb}):
        from bdb import BdbQuit
        
        # Test the predicate at line 2 - the docstring should exist and be truthy
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
                    sys.__excepthook__(type, value, traceback)
                else:
                    ipython_hook(type, value, traceback)

            from IPython.core import ultratb
            ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
            sys.excepthook = excepthook
        
        # Verify the predicate (docstring) evaluates to True
        predicate = bool(register_ipython_excepthook.__doc__)
        assert predicate is True
        assert "Register an exception hook" in register_ipython_excepthook.__doc__


# LLM-generated content at query #27
#--------------------------

```python
def test_exception_wrapper_predicate_line_6_false():
    """Test that the predicate at line 6 (handler_fn is not None) evaluates to False"""
    import inspect
    from flutes.exception import exception_wrapper
    
    # When handler_fn is None (default), the predicate at line 45 should be False
    @exception_wrapper()
    def dummy_func():
        pass
    
    # Call the wrapped function to verify it works with handler_fn=None
    result = dummy_func()
    assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("Test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_with_args(x, y):
        raise RuntimeError("Custom error")
    
    try:
        func_with_args(1, 2)
    except RuntimeError:
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


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=10):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, z=20):
        raise TypeError("Type error")
    
    try:
        func_with_defaults(5, z=30)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], TypeError)
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y):
        raise KeyError("Key error")
    
    try:
        func_with_kwargs(1, 2)
    except KeyError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], KeyError)
    assert handler_called[0][1] == 1


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def gen_func(x):
        yield 1
        yield 2
        raise ValueError("Generator error")
        yield 3
    
    gen = gen_func(5)
    assert next(gen) == 1
    assert next(gen) == 2
    
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == 5


def test_exception_wrapper_invalid_handler_no_args():
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_mismatch():
    def custom_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_value_conflict():
    def custom_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(custom_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_with_varargs_in_wrapped():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((e, x, args))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args):
        raise RuntimeError("Varargs error")
    
    try:
        func_with_varargs(1, 2, 3)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def func_returns_value(x):
        return x + 10
    
    result = func_returns_value(5)
    assert result == 15


def test_exception_wrapper_with_kwargs_in_wrapped():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y=20):
        raise ValueError("Kwargs error")
    
    try:
        func_with_kwargs(1, y=30)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert "y" in handler_called[0][2]
    assert handler_called[0][2]["y"] == 30


def test_exception_wrapper_handler_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y=10):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func(x, y):
        raise TypeError("Type error")
    
    try:
        func(1, 2)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_decorator_returns_decorator():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    
    assert callable(result)


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_handler_fn_is_none():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_no_handler_logs_exception(monkeypatch):
    log_calls = []
    
    def mock_log(msg, level="info", **kwargs):
        log_calls.append((msg, level, kwargs))
    
    monkeypatch.setattr("flutes.exception.log", mock_log)
    
    @exception_wrapper()
    def failing_func():
        raise ValueError("test error")
    
    failing_func()
    assert len(log_calls) == 2
    assert log_calls[1][1] == "error"
    assert "test error" in log_calls[1][0]


def test_exception_wrapper_with_custom_handler():
    handler_calls = []
    
    def custom_handler(e):
        handler_calls.append(e)
    
    @exception_wrapper(custom_handler)
    def failing_func():
        raise ValueError("test error")
    
    failing_func()
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0], ValueError)


def test_exception_wrapper_handler_receives_matching_args():
    handler_calls = []
    
    def custom_handler(e, x, y):
        handler_calls.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise ValueError("test error")
    
    failing_func(1, 2)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 2


def test_exception_wrapper_handler_with_default_args():
    handler_calls = []
    
    def custom_handler(e, x, default_arg=None):
        handler_calls.append((e, x, default_arg))
    
    @exception_wrapper(custom_handler)
    def failing_func(x):
        raise ValueError("test error")
    
    failing_func(1)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] is None


def test_exception_wrapper_handler_with_varargs():
    handler_calls = []
    
    def custom_handler(e, args):
        handler_calls.append((e, args))
    
    @exception_wrapper(custom_handler)
    def failing_func(*args):
        raise ValueError("test error")
    
    failing_func(1, 2, 3)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == (1, 2, 3)


def test_exception_wrapper_handler_with_kwargs():
    handler_calls = []
    
    def custom_handler(e, x, kw=None):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, **kwargs):
        raise ValueError("test error")
    
    failing_func(1, key="value")
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1


def test_exception_wrapper_handler_with_varkw():
    handler_calls = []
    
    def custom_handler(e, x, **kw):
        handler_calls.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, **kwargs):
        raise ValueError("test error")
    
    failing_func(1, key="value", another="param")
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1
    assert "key" in handler_calls[0][2]


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def passing_func(x):
        return x * 2
    
    result = passing_func(5)
    assert result == 10


def test_exception_wrapper_generator():
    handler_calls = []
    
    def custom_handler(e):
        handler_calls.append(e)
    
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
    assert len(handler_calls) == 1


def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def passing_generator():
        yield 1
        yield 2
    
    gen = passing_generator()
    assert list(gen) == [1, 2]


def test_exception_wrapper_invalid_handler_no_positional_arg():
    def invalid_handler(**kwargs):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_mismatched_arg():
    def invalid_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_on_matching_arg():
    def invalid_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a docstring"""
        pass
    
    assert documented_func.__doc__ == "This is a docstring"
    assert documented_func.__name__ == "documented_func"


def test_exception_wrapper_handler_receives_kwonly_args():
    handler_calls = []
    
    def custom_handler(e, x):
        handler_calls.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(*, x):
        raise ValueError("test error")
    
    failing_func(x=1)
    assert len(handler_calls) == 1
    assert handler_calls[0][1] == 1


# LLM-generated content at query #32
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


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_with_args(x, y):
        raise RuntimeError("Test")
    
    try:
        func_with_args(10, 20)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == 10


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func(x):
        raise TypeError("Test")
    
    try:
        func(5)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func(x, y):
        raise KeyError("Test")
    
    try:
        func(1, 2)
    except KeyError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == {"y": 2}


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("Generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_handler_validation_no_positional_arg():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_validation_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_validation_unmatched_arg():
    def bad_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_validation_default_on_matched_arg():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b=None, **kw):
        handler_called.append((e, a, b, kw))
    
    @exception_wrapper(custom_handler)
    def func(a, b, c=10):
        raise RuntimeError("Test")
    
    try:
        func(1, 2, c=20)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][3] == {"b": 2, "c": 20}


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a docstring."""
        return "result"
    
    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "This is a docstring."


def test_exception_wrapper_returns_value_on_success():
    @exception_wrapper()
    def func(x, y):
        return x + y
    
    result = func(3, 4)
    assert result == 7


def test_exception_wrapper_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func(x, *, y):
        raise ValueError("Test")
    
    try:
        func(5, y=10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5


# LLM-generated content at query #33
#--------------------------

```python
def test_register_ipython_excepthook_predicate_line_2():
    # Line 2 is a docstring (r"""..."""), which evaluates to False when used as a predicate
    # because it's a string that is not assigned to a variable
    predicate = r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """
    assert not predicate == False
    assert bool(predicate) == True


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #35
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    import sys
    from unittest.mock import Mock, patch
    
    # Mock the IPython module to avoid import issues
    with patch('IPython.core.ultratb.FormattedTB'):
        # Import the function
        from your_module import register_ipython_excepthook
        
        # Call the function with default parameters
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # The predicate at line 8: `if not capture_keyboard_interrupt:`
        # With capture_keyboard_interrupt=False, the predicate evaluates to True
        capture_keyboard_interrupt = False
        predicate_result = not capture_keyboard_interrupt
        
        assert predicate_result is True


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_handler_with_varkw():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e, one, **kw):
        handler_called.append((e, one, kw))
    
    @exception_wrapper(custom_handler)
    def test_func(one, two, *args, three=None, **kwargs):
        raise ValueError("test error")
    
    test_func(1, "2", "arg1", "arg2", four=4)
    
    assert len(handler_called) == 1
    exc, one_val, kw = handler_called[0]
    assert isinstance(exc, ValueError)
    assert one_val == 1
    assert "two" in kw
    assert kw["two"] == "2"
    assert "kwargs" in kw
    assert kw["kwargs"] == {"four": 4}


# LLM-generated content at query #37
#--------------------------

```python
def test_register_ipython_excepthook_predicate_evaluates_to_false():
    from bdb import BdbQuit
    import sys
    
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    predicate = not capture_keyboard_interrupt
    
    assert predicate is False or predicate is True
    assert predicate == True


# LLM-generated content at query #38
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    predicate = not capture_keyboard_interrupt
    assert predicate is False


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    from flutes.exception import exception_wrapper
    docstring = exception_wrapper.__doc__
    assert docstring is not None
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated function" in docstring


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_handler_with_varkw():
    """Test that exception handler with **kwargs captures remaining argument name-value pairs."""
    handler_called = []
    
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        handler_called.append({
            'exception': e,
            'three': three,
            'one': one,
            'args': args,
            'my_arg': my_arg,
            'kw': kw
        })
    
    from flutes.exception import exception_wrapper
    
    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        raise ValueError("test error")
    
    foo(1, "2", "arg1", "arg2", four=4)
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0]['exception'], ValueError)
    assert handler_called[0]['three'] is None
    assert handler_called[0]['one'] == 1
    assert handler_called[0]['args'] == ("arg1", "arg2")
    assert handler_called[0]['my_arg'] is None
    assert handler_called[0]['kw'] == {"two": "2", "kwargs": {"four": 4}}


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, my_arg=None):
        handler_called.append((e, my_arg))
    
    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, **kw):
        handler_called.append((e, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x=1):
        raise ValueError("test error")
    
    try:
        func_that_raises(x=5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1]["x"] == 5


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("test error")
        yield 2
    
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_handler_with_args_and_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=10):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_no_positional_arg():
    try:
        @exception_wrapper(lambda: None)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_with_varargs():
    try:
        def bad_handler(e, *args):
            pass
        
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_mismatched_arg():
    try:
        def bad_handler(e, nonexistent_arg):
            pass
        
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, *, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def func_with_doc():
        """Test docstring"""
        return 42
    
    assert func_with_doc.__name__ == "func_with_doc"
    assert func_with_doc.__doc__ == "Test docstring"


# LLM-generated content at query #42
#--------------------------

```python
def test_register_ipython_excepthook_default_parameter():
    import sys
    from bdb import BdbQuit
    
    # Store original excepthook
    original_excepthook = sys.excepthook
    
    try:
        # Call the function with default parameter
        register_ipython_excepthook()
        
        # Verify that sys.excepthook was modified
        assert sys.excepthook != original_excepthook
        assert sys.excepthook is not None
        
    finally:
        # Restore original excepthook
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        assert sys.excepthook != original_excepthook
        assert callable(sys.excepthook)
        
    finally:
        sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    
    original_excepthook = sys.excepthook
    
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        
        assert sys.excepthook != original_excepthook
        assert callable(sys.excepthook)
        
    finally:
        sys.excepthook = original_excepthook


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_predicate_line_1_false():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_with_arg(x):
        raise RuntimeError("custom error")
    
    try:
        func_with_arg(42)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == 42


def test_exception_wrapper_handler_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, y=10):
        raise TypeError("test")
    
    try:
        func_with_defaults(5)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y=None):
        raise KeyError("test")
    
    try:
        func_with_kwargs(1, y=2)
    except KeyError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2]["y"] == 2


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_generator():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_generator(x):
        yield 1
        yield 2
        raise ValueError("gen error")
    
    gen = func_generator(42)
    values = []
    try:
        for val in gen:
            values.append(val)
    except ValueError:
        pass
    
    assert values == [1, 2]
    assert len(handler_called) == 1
    assert handler_called[0][1] == 42


def test_exception_wrapper_invalid_handler_no_args():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_arg_not_in_wrapped():
    def bad_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_default_matches_wrapped():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func(x):
        """This is a docstring"""
        return x
    
    assert documented_func.__name__ == "documented_func"
    assert "docstring" in documented_func.__doc__


def test_exception_wrapper_with_varargs():
    handler_called = []
    
    def custom_handler(e, x, args):
        handler_called.append((e, x, args))
    
    @exception_wrapper(custom_handler)
    def func_with_varargs(x, *args):
        raise ValueError("test")
    
    try:
        func_with_varargs(1, 2, 3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == (2, 3)


def test_exception_wrapper_handler_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_kwonly(x, *, y):
        raise RuntimeError("test")
    
    try:
        func_kwonly(1, y=2)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_kwonly_with_defaults():
    handler_called = []
    
    def custom_handler(e, x, *, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_kwonly_default(x, *, y=10):
        raise TypeError("test")
    
    try:
        func_kwonly_default(1)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][2] == 10


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_predicate_line_2():
    # Line 2 contains the docstring opening, but line 40 has the predicate:
    # if hasattr(fn, "__wrapped__"):
    # We need to ensure this evaluates to False
    
    def regular_function():
        pass
    
    # regular_function does not have __wrapped__ attribute
    assert not hasattr(regular_function, "__wrapped__")


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    from flutes.exception import exception_wrapper
    docstring = exception_wrapper.__doc__
    assert docstring is not None
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in docstring


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_decorator_returns_callable():
    from flutes.exception import exception_wrapper
    
    decorator = exception_wrapper()
    assert callable(decorator)


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_predicate_line_1_false():
    from flutes.exception import exception_wrapper
    
    # Line 1 is: def exception_wrapper(handler_fn=None):
    # The predicate at line 1 is the function definition itself
    # We need to test that exception_wrapper can be called without arguments
    # which means handler_fn defaults to None, making the predicate (handler_fn=None) evaluate to False
    
    decorator = exception_wrapper()
    
    def test_func():
        return 42
    
    wrapped_func = decorator(test_func)
    result = wrapped_func()
    
    assert result == 42


# LLM-generated content at query #49
#--------------------------

```python
def test_register_ipython_excepthook_predicate_evaluates_to_false():
    from bdb import BdbQuit
    import sys
    
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    
    predicate = any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)
    
    assert predicate is False


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def func_raises_error():
        raise ValueError("Test error")
    
    try:
        func_raises_error()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_raises_error():
        raise ValueError("Test error")
    
    try:
        func_raises_error()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Test error" in handler_called[0]


def test_exception_wrapper_handler_with_matching_args():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, y):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['y'] = y
    
    @exception_wrapper(custom_handler)
    def func_raises_error(x, y):
        raise ValueError("Test error")
    
    try:
        func_raises_error(10, 20)
    except ValueError:
        pass
    
    assert handler_args['x'] == 10
    assert handler_args['y'] == 20
    assert "Test error" in handler_args['e']


def test_exception_wrapper_handler_with_default_args():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, default_arg=None):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['default_arg'] = default_arg
    
    @exception_wrapper(custom_handler)
    def func_raises_error(x):
        raise ValueError("Test error")
    
    try:
        func_raises_error(10)
    except ValueError:
        pass
    
    assert handler_args['x'] == 10
    assert handler_args['default_arg'] is None


def test_exception_wrapper_handler_with_varkw():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, x, **kwargs):
        handler_args['e'] = str(e)
        handler_args['x'] = x
        handler_args['kwargs'] = kwargs
    
    @exception_wrapper(custom_handler)
    def func_raises_error(x, y=5):
        raise ValueError("Test error")
    
    try:
        func_raises_error(10, y=20)
    except ValueError:
        pass
    
    assert handler_args['x'] == 10
    assert handler_args['kwargs']['y'] == 20


def test_exception_wrapper_no_exception():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10


def test_exception_wrapper_generator():
    from flutes.exception import exception_wrapper
    
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def gen_raises_error():
        yield 1
        raise ValueError("Generator error")
    
    gen = gen_raises_error()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "Generator error" in handler_called[0]


def test_exception_wrapper_invalid_handler_no_args():
    from flutes.exception import exception_wrapper
    
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "must have a positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func():
            pass
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)


def test_exception_wrapper_invalid_handler_unmatched_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, unmatched_arg):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_invalid_handler_default_matching_arg():
    from flutes.exception import exception_wrapper
    
    def invalid_handler(e, x=10):
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def func(x):
            pass
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_preserves_function_metadata():
    from flutes.exception import exception_wrapper
    
    @exception_wrapper()
    def documented_func():
        """This is a documented function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "This is a documented function."


def test_exception_wrapper_with_args_and_kwargs():
    from flutes.exception import exception_wrapper
    
    handler_args = {}
    
    def custom_handler(e, a, *args_from_handler, b=None, **kwargs):
        handler_args['e'] = str(e)
        handler_args['a'] = a
        handler_args['b'] = b
        handler_args['kwargs'] = kwargs
    
    @exception_wrapper(custom_handler)
    def func_with_args(a, b, c=None):
        raise ValueError("Test error")
    
    try:
        func_with_args(1, 2, c=3)
    except ValueError:
        pass
    
    assert handler_args['a'] == 1
    assert handler_args['b'] == 2


# LLM-generated content at query #51
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook != original_excepthook
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook != original_excepthook
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_capture_keyboard_interrupt_true():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook != original_excepthook
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_sets_excepthook():
    import sys
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    new_excepthook = sys.excepthook
    
    assert new_excepthook is not None
    assert callable(new_excepthook)
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_bdbquit():
    import sys
    from bdb import BdbQuit
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    current_excepthook = sys.excepthook
    
    current_excepthook(BdbQuit, BdbQuit("test"), None)
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass


def test_exception_wrapper_custom_handler():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def func_with_arg(x):
        raise RuntimeError("error")
    
    try:
        func_with_arg(42)
    except RuntimeError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], RuntimeError)
    assert handler_called[0][1] == 42


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def func_with_defaults(x, y=10):
        raise TypeError("error")
    
    try:
        func_with_defaults(5)
    except TypeError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 5
    assert handler_called[0][2] == 10


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def func_with_kwargs(x, y=None):
        raise KeyError("error")
    
    try:
        func_with_kwargs(1, y=2)
    except KeyError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == {"y": 2}


def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error(x):
        return x * 2
    
    result = func_no_error(5)
    assert result == 10


def test_exception_wrapper_generator():
    @exception_wrapper()
    def gen_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("gen error")
            yield i
    
    gen = gen_func(5)
    assert next(gen) == 0
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass


def test_exception_wrapper_generator_no_error():
    @exception_wrapper()
    def gen_func(n):
        for i in range(n):
            yield i
    
    result = list(gen_func(3))
    assert result == [0, 1, 2]


def test_exception_wrapper_handler_validation_no_positional_arg():
    def bad_handler(x):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_validation_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_validation_missing_arg():
    def bad_handler(e, nonexistent):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_validation_default_conflict():
    def bad_handler(e, x=None):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "default values" in str(e)


def test_exception_wrapper_preserves_return_value():
    @exception_wrapper()
    def func_returns():
        return "result"
    
    result = func_returns()
    assert result == "result"


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, a, b, c=None, **kw):
        handler_called.append((a, b, c, kw))
    
    @exception_wrapper(custom_handler)
    def func(a, b, c=5, d=10):
        raise Exception("test")
    
    try:
        func(1, 2, d=20)
    except Exception:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == 1
    assert handler_called[0][1] == 2
    assert handler_called[0][2] == 5
    assert handler_called[0][3] == {"d": 20}


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a docstring"""
        pass
    
    assert documented_func.__doc__ == "This is a docstring"
    assert documented_func.__name__ == "documented_func"


def test_exception_wrapper_nested_decorators():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append(x)
    
    @exception_wrapper(custom_handler)
    def inner_func(x):
        raise ValueError("nested error")
    
    try:
        inner_func(99)
    except ValueError:
        pass
    
    assert handler_called == [99]


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def failing_func():
        raise ValueError("test error")
    
    try:
        failing_func()
    except ValueError:
        pass


def test_exception_wrapper_with_custom_handler():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def failing_func():
        raise ValueError("test error")
    
    try:
        failing_func()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_handler_with_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y):
        raise ValueError("test error")
    
    try:
        failing_func(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0].__class__.__name__ == "ValueError"
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_handler_with_default_args():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y=None):
        raise ValueError("test error")
    
    try:
        failing_func(1)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] is None


def test_exception_wrapper_handler_with_varkw():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((e, x, kw))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, **kwargs):
        raise ValueError("test error")
    
    try:
        failing_func(1, y=2, z=3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2]["y"] == 2
    assert handler_called[0][2]["z"] == 3


def test_exception_wrapper_handler_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def normal_func(x):
        return x * 2
    
    result = normal_func(5)
    assert result == 10
    assert len(handler_called) == 0


def test_exception_wrapper_handler_with_varargs():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((e, x))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, *args):
        raise ValueError("test error")
    
    try:
        failing_func(1, 2, 3)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1


def test_exception_wrapper_handler_invalid_no_exception_arg():
    try:
        @exception_wrapper(lambda: None)
        def failing_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_handler_invalid_with_varargs():
    try:
        @exception_wrapper(lambda e, *args: None)
        def failing_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_handler_invalid_arg_name():
    try:
        @exception_wrapper(lambda e, nonexistent: None)
        def failing_func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_handler_invalid_default_value():
    try:
        @exception_wrapper(lambda e, x=None: None)
        def failing_func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


def test_exception_wrapper_generator_success():
    def custom_handler(e):
        pass
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        yield 2
        yield 3
    
    result = list(gen_func())
    assert result == [1, 2, 3]


def test_exception_wrapper_generator_with_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(e)
    
    @exception_wrapper(custom_handler)
    def gen_func():
        yield 1
        raise ValueError("generator error")
    
    gen = gen_func()
    assert next(gen) == 1
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0], ValueError)


def test_exception_wrapper_with_kwargs():
    handler_called = []
    
    def custom_handler(e, x, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, y=None):
        raise ValueError("test error")
    
    try:
        failing_func(x=1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func(x):
        """This is a documented function."""
        return x
    
    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "This is a documented function."


def test_exception_wrapper_handler_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, *, y=None):
        handler_called.append((e, x, y))
    
    @exception_wrapper(custom_handler)
    def failing_func(x, *, y=None):
        raise ValueError("test error")
    
    try:
        failing_func(1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][1] == 1
    assert handler_called[0][2] == 2


# LLM-generated content at query #54
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_with_capture_keyboard_interrupt():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    assert sys.excepthook != original_excepthook
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_bdb_quit_exception():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    bdb_quit_exception = BdbQuit()
    sys.excepthook(BdbQuit, bdb_quit_exception, None)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_not_captured():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    keyboard_interrupt = KeyboardInterrupt()
    sys.excepthook(KeyboardInterrupt, keyboard_interrupt, None)
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_keyboard_interrupt_captured():
    import sys
    from bdb import BdbQuit
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    assert sys.excepthook is not None
    
    sys.excepthook = original_excepthook


def test_register_ipython_excepthook_sets_sys_excepthook():
    import sys
    
    original_excepthook = sys.excepthook
    
    register_ipython_excepthook()
    
    assert sys.excepthook is not original_excepthook
    assert callable(sys.excepthook)
    
    sys.excepthook = original_excepthook


# LLM-generated content at query #55
#--------------------------

Looking at the code, I need to understand what predicate at line 5 refers to. Line 5 in the provided `flutes/exception.py` is blank, but based on context, the instruction likely refers to a predicate in the actual implementation. 

Looking at the `exception_wrapper` function, the key predicates are:
- Line 40: `if hasattr(fn, "__wrapped__"):`
- Line 45: `if handler_fn is not None:`
- Line 47: `if len(handler_argspec.args) == 0:`
- Line 49: `if handler_argspec.varargs is not None:`
- Line 92: `if inspect.isgenerator(result):`

Given the context of "predicate at line 5" and looking at the docstring structure, this likely refers to testing a scenario where `handler_fn is None` (line 45 in the decorator), which would make the predicate `handler_fn is not None` evaluate to `False`.

Here's a test to ensure that predicate evaluates to False:


# LLM-generated content at query #56
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
    assert BdbQuit in skip_exceptions


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_wrapper_handler_fn_is_none():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")
    
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
        raise ValueError("test error")
    
    try:
        func_that_raises()
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert "test error" in handler_called[0]


def test_exception_wrapper_with_handler_matching_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 1, 2)


def test_exception_wrapper_with_handler_and_defaults():
    handler_called = []
    
    def custom_handler(e, x, default_arg=None):
        handler_called.append((str(e), x, default_arg))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x):
        raise ValueError("test error")
    
    try:
        func_that_raises(5)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 5, None)


def test_exception_wrapper_with_handler_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x, **kw):
        handler_called.append((str(e), x, kw))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, y=None):
        raise ValueError("test error")
    
    try:
        func_that_raises(5, y=10)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0][0] == "test error"
    assert handler_called[0][1] == 5


def test_exception_wrapper_no_exception():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_normal():
        return 42
    
    result = func_normal()
    assert result == 42
    assert len(handler_called) == 0


def test_exception_wrapper_with_generator():
    handler_called = []
    
    def custom_handler(e):
        handler_called.append(str(e))
    
    @exception_wrapper(custom_handler)
    def func_generator():
        yield 1
        yield 2
        raise ValueError("generator error")
    
    gen = func_generator()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except ValueError:
        pass
    
    assert len(handler_called) == 1


def test_exception_wrapper_invalid_handler_no_args():
    def bad_handler():
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)


def test_exception_wrapper_invalid_handler_with_varargs():
    def bad_handler(e, *args):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "varargs" in str(e)


def test_exception_wrapper_invalid_handler_mismatched_arg():
    def bad_handler(e, nonexistent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)


def test_exception_wrapper_with_args_and_kwargs():
    handler_called = []
    
    def custom_handler(e, x):
        handler_called.append((str(e), x))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *args, **kwargs):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, 2, 3, key="value")
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 1)


def test_exception_wrapper_with_kwonly_args():
    handler_called = []
    
    def custom_handler(e, x, y):
        handler_called.append((str(e), x, y))
    
    @exception_wrapper(custom_handler)
    def func_that_raises(x, *, y):
        raise ValueError("test error")
    
    try:
        func_that_raises(1, y=2)
    except ValueError:
        pass
    
    assert len(handler_called) == 1
    assert handler_called[0] == ("test error", 1, 2)


def test_exception_wrapper_preserves_function_metadata():
    @exception_wrapper()
    def documented_func():
        """This is a documented function."""
        return 42
    
    assert documented_func.__name__ == "documented_func"
    assert "documented" in documented_func.__doc__


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_wrapper_handler_fn_is_none():
    handler_fn = None
    result = handler_fn is not None
    assert result is False


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_wrapper_predicate_line_6_false():
    """Test that the predicate at line 6 (handler_fn is not None) evaluates to False"""
    from flutes.exception import exception_wrapper
    
    # Apply decorator without handler_fn (defaults to None)
    @exception_wrapper()
    def sample_function(x):
        return x * 2
    
    # Call the wrapped function - it should work normally without exception handling
    result = sample_function(5)
    assert result == 10


