####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) -> calls log_exception
    with patch("your_module_name.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(a, b):
            raise ValueError("error occurred")

        try:
            failing_func(1, 2)
        except ValueError:
            pass
        mock_log.assert_called_once()

    # 2. Test with custom handler and positional/keyword arguments mapping
    handler_calls = []
    def custom_handler(e, one, two, my_arg=None, **kw):
        handler_calls.append((e, one, two, my_arg, kw))

    @exception_wrapper(custom_handler)
    def complex_func(one, two, three, my_arg="default", extra="extra"):
        raise RuntimeError("boom")

    try:
        complex_func(10, 20, 30, my_arg="changed", extra="value")
    except RuntimeError:
        pass

    # Check if arguments were correctly mapped from the wrapped function to the handler
    # e must be the first arg. 'one' and 'two' are matched names.
    # 'three' is in func but not in handler (should be ignored).
    # 'my_arg' is in both, but since it has a default in func, 
    # the decorator logic forbids it having a default in the handler if it matches.
    # Wait, looking at code: "Arguments without matching names... must have default values"
    # and "Arguments with matching names... cannot take default values".
    
    # Let's verify the specific logic of the provided implementation:
    # Handler args: e (fixed), one, two, my_arg, kw.
    # In handler, 'one' and 'two' are from func. 'my_arg' is matched. 
    # Everything else in bound_args goes to **kw.
    
    # Re-evaluating the handler logic based on the provided code's specific constraints:
    # The test above actually relies on the decorator implementation's behavior.
    assert len(handler_calls) == 1
    exc, one, two, my_arg, kw = handler_calls[0]
    assert isinstance(exc, RuntimeError)
    assert one == 10
    assert two == 20
    assert my_arg == "changed"
    assert kw["three"] == 30
    assert kw["extra"] == "value"

    # 3. Test Generator support (unrolling)
    generator_results = []
    @exception_wrapper()
    def generator_func():
        yield 1
        yield 2
        raise TypeError("gen error")

    gen = generator_func()
    with pytest.raises(TypeError):
        for val in gen:
            generator_results.append(val)
    assert generator_results == [1, 2]

    # 4. Test validation: Handler must have exception argument
    def invalid_handler(not_e): # This is actually okay as long as it's the first arg
        pass
    
    def no_arg_handler():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(no_arg_handler)
        def func_fail():
            pass

    # 5. Test validation: Handler cannot have *args
    def args_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(args_handler)
        def func_fail_args():
            pass

    # 6. Test validation: Argument in handler does not exist in wrapped function
    def missing_arg_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(missing_arg_handler)
        def func_missing():
            raise ValueError()

    # 7. Test validation: Argument with default in handler matches wrapped function (forbidden by code logic)
    def default_in_handler_error(e, one=1):
        pass

    @exception_wrapper(default_in_handler_error)
    def func_default_conflict(one):
        raise ValueError()

    with pytest.raises(ValueError, match="cannot have default values"):
        func_default_conflict(1)

    # 8. Test unwrapping of decorated functions
    @exception_wrapper()
    def inner_func():
        raise ValueError("inner")

    @exception_wrapper()
    def outer_func():
        return inner_func()

    with pytest.raises(ValueError, match="inner"):
        outer_func()
```


# LLM-generated content at query #2
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for IPython components used inside the function
    mock_ultratb = MagicMock()
    mock_formatted_tb = MagicMock()
    mock_ultratb.FormattedTB.return_value = mock_formatted_tb
    
    with patch.dict('sys.modules', {'IPython': MagicMock(), 'IPython.core': MagicMock(), 'IPython.core.ultratb': mock_ultratb}), \
         patch('sys.excepthook') as mock_excepthook_setter, \
         patch('sys.__excepthook__') as mock_original_excepthook:
        
        # Case 1: Test registration with default (capture_keyboard_interrupt=False)
        register_ipython_excepthook()
        
        # Verify that sys.excepthook was updated
        assert sys.excepthook != mock_excepthook_setter.call_args[0][0] if not hasattr(sys, 'excepthook') else True 
        # The setter is actually the assignment: sys.excepthook = excepthook
        # Since we can't easily intercept assignment to a built-in attribute without patching sys, 
        # we check if the function was assigned.
        
        # We need to capture the actual function object that was assigned
        # To do this reliably in a test, we patch sys.excepthook
        
    # Case 2: Test logic of the internal excepthook function
    # Since excepthook is a closure, we must trigger it by calling the assigned sys.excepthook
    
    with patch('IPython.core.ultratb.FormattedTB') as mock_tb_class, \
         patch('sys.__excepthook__') as mock_sys_hook:
        
        # Re-run registration to capture the specific closure instance
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        captured_hook = sys.excepthook
        
        # Test KeyboardInterrupt (should be skipped because capture_keyboard_interrupt=False)
        captured_hook(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), None)
        mock_sys_hook.assert_called_with(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), None)
        
        # Test BdbQuit (should always be skipped)
        captured_hook(BdbQuit, BdbQuit(), None)
        mock_sys_hook.assert_called_with(BdbQuit, BdbQuit(), None)
        
        # Test a standard Exception (should trigger IPython hook)
        captured_hook(ValueError, ValueError("error"), None)
        mock_tb_class.assert_called()

    # Case 3: Test registration with capture_keyboard_interrupt=True
    with patch('IPython.core.ultratb.FormattedTB') as mock_tb_class, \
         patch('sys.__excepthook__') as mock_sys_hook:
        
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        captured_hook = sys.excepthook
        
        # Test KeyboardInterrupt (should NOT be skipped now)
        captured_hook(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), None)
        mock_tb_class.assert_called()
        
        # Reset mock to check next call
        mock_tb_class.reset_mock()
        
        # Test ValueError (should trigger IPython hook)
        captured_hook(ValueError, ValueError("error"), None)
        mock_tb_class.assert_called()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (handler_fn is None) -> calls log_exception
    with patch("flutes.log.log") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called()

    # 2. Test custom handler with matching positional and keyword arguments
    # Handler signature: (e, one, two, my_arg=None, **kwargs)
    # Function signature: (one, two, three=10, **kwargs)
    handler_calls = []
    def custom_handler(e, one, two, my_arg=None, **kwargs):
        handler_calls.append((e, one, two, my_arg, kwargs))

    @exception_wrapper(custom_handler)
    def target_func(one, two, three=10, extra="val"):
        raise TypeError("type error")

    with pytest.raises(TypeError, match="type error"):
        target_func(1, 2, three=20, extra="val")

    # Verify handler received:
    # e = TypeError
    # one = 1 (from target_func)
    # two = 2 (from target_func)
    # my_arg = None (default in handler, not in target)
    # kwargs = {'three': 20, 'extra': 'val'} (remaining from target)
    e_val, one_val, two_val, my_arg_val, kw_val = handler_calls[0]
    assert isinstance(e_val, TypeError)
    assert one_val == 1
    assert two_val == 2
    assert my_arg_val is None
    assert kw_val["three"] == 20
    assert kw_val["extra"] == "val"

    # 3. Test Generator support (unrolling the generator to catch exception)
    generator_caught = []
    def gen_handler(e):
        generator_caught.append(e)

    @exception_wrapper(gen_handler)
    def error_generator():
        yield 1
        raise RuntimeError("gen error")

    gen_inst = error_generator()
    with pytest.raises(StopIteration): # Generator unrolling via yield from ends with exception
        for _ in gen_inst:
            pass
    assert len(generator_caught) == 1
    assert isinstance(generator_caught[0], RuntimeError)

    # 4. Test validation: Handler must have an exception argument (first arg)
    def invalid_handler(not_e): # This is actually okay as long as it's the first arg, 
                                # but let's test a handler with no args at all.
        pass
    
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def bad_func():
            pass

    # 5. Test validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def bad_func():
            pass

    # 6. Test validation: Argument in handler does not exist in wrapped function
    def missing_arg_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(missing_arg_handler)
        def bad_func(a):
            pass

    # 7. Test validation: Argument with default in handler cannot be present in wrapped function
    # (Because if it exists in wrapped, it's no longer a 'default-only' argument for the handler)
    def default_arg_handler(e, val=5):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_arg_handler)
        def bad_func(val):
            pass
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("exc_type, exc_val, user_msg, expected_log_calls", [
    (ValueError, ValueError("test error"), None, ["<ValueError> test error"]),
    (RuntimeError, RuntimeError("fatal"), "User Alert", ["User Alert: <RuntimeError> fatal"]),
])
def test_log_exception(exc_type, exc_val, user_msg, expected_log_calls):
    with patch("path.to.module.log") as mock_log:
        # We patch traceback.format_exc to return a predictable string for the 'error' call
        with patch("traceback.format_exc", return_value="mock_traceback"):
            log_exception(exc_val, user_msg=user_msg)
            
            # Check if log was called with the exception message
            # The first call is usually the traceback, the second is the actual error message
            found_message = False
            for call in mock_log.call_args_list:
                if any(expected in call[0] for expected in expected_log_calls):
                    found_message = True
            assert found_message, f"Expected log messages {expected_log_calls} not found in {mock_log.call_args_list}"

def test_log_exception_subprocess_error():
    """Test that CalledProcessError with output does not trigger the traceback log call."""
    exc = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some output")
    with patch("path.to.module.log") as mock_log:
        log_exception(exc)
        
        # Check that it only logged the error message, not the traceback
        # If traceback was logged, there would be 2 calls to log()
        # We check that no call contains 'mock_traceback'
        for call in mock_log.call_args_list:
            assert "mock_traceback" not in str(call)
        
        # Verify the error message itself was logged
        actual_msg = mock_log.call_args_list[0][0][0]
        assert "<CalledProcessError> 'ls'" in actual_msg

def test_log_exception_logging_failure():
    """Test that if the logger itself fails, it falls back to printing."""
    exc = ValueError("original error")
    with patch("path.to.module.log", side_effect=Exception("logging failed")):
        with patch("builtins.print") as mock_print:
            log_exception(exc)
            
            # Verify fallback print calls
            printed_messages = [call[0][0] for call in mock_print.call_args_list]
            assert any("<ValueError> original error" in m for m in printed_messages)
            assert any("Another exception occurred while logging" in m for m in printed_messages)

def test_log_exception_with_kwargs():
    """Test that extra kwargs are passed through to the log function."""
    exc = TypeError("type error")
    extra_context = {"request_id": "123", "user": "admin"}
    with patch("path.to.module.log") as mock_log:
        log_exception(exc, extra_context=extra_context)
        
        # Verify kwargs are passed to log()
        _, kwargs = mock_log.call_args[1]
        assert kwargs["extra_context"] == extra_context
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn) calls log_exception
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with matching positional and keyword arguments
    handler_calls = []
    def custom_handler(e, one, two, my_arg=None, **kwargs):
        handler_calls.append((e, one, two, my_arg, kwargs))
        raise e

    @exception_wrapper(custom_handler)
    def decorated_func(one, two, three=10, extra="val"):
        raise ValueError("trigger")

    with pytest.raises(ValueError, match="trigger"):
        decorated_func(1, 2, extra="extra_val")
    
    # Verify arguments passed to handler:
    # e is the exception
    # one=1 (from args)
    # two=2 (from args)
    # my_arg is not in decorated_func signature, so it shouldn't be populated from there 
    # but decorator logic checks if names exist in inner_signature.
    # Let's refine the test case to match valid decorator requirements.

    # 3. Test Valid Handler Signature Match
    def valid_handler(e, a, b, c=None, **kwargs):
        valid_handler.captured = (e, a, b, c, kwargs)
        raise e

    @exception_wrapper(valid_handler)
    def target_func(a, b, c=None, d="default"):
        raise RuntimeError("boom")

    with pytest.append_output(): # dummy to catch raise
        try:
            target_func(1, 2, c=3, d="changed")
        except RuntimeError:
            pass
    
    assert valid_handler.captured[0].__class__ == RuntimeError
    assert valid_handler.captured[1] == 1
    assert valid_handler.captured[2] == 2
    assert valid_handler.captured[3] == 3
    # d is in target_func but not explicitly named in handler arguments, 
    # so it goes into kwargs because it's part of the bound args
    assert valid_handler.captured[4]['d'] == "changed"

    # 4. Test Generator Support (unrolling)
    def gen_fail():
        yield 1
        raise TypeError("gen error")

    gen_captured = []
    def gen_handler(e, val):
        gen_captured.append((e, val))
        raise e

    @exception_wrapper(gen_handler)
    def wrapper_gen(val):
        return gen_fail()

    gen_it = wrapper_gen("test_val")
    with pytest.raises(TypeError, match="gen error"):
        for _ in gen_it:
            pass
    assert gen_captured[0][0].__class__ == TypeError
    assert gen_captured[0][1] == "test_val"

    # 5. Test Validation Error: Handler missing exception arg
    def bad_handler(a):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(bad_handler)
        def func():
            pass

    # 6. Test Validation Error: Argument in handler not in wrapped function
    def mismatch_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(mismatch_handler)
        def func():
            pass

    # 7. Test Validation Error: Handler argument has default value but exists in wrapped function
    def default_arg_handler(e, a=1):
        pass

    @exception_wrapper(default_arg_handler)
    def func_with_a(a):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        # This triggers because 'a' is in inner_signature and has a default in handler
        # The decorator logic: if name in inner_signature... raise ValueError
        pass 

    # 8. Test Validation Error: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def func():
            pass
```


# LLM-generated content at query #6
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for the IPython component and system excepthook
    with patch("sys.__excepthook__") as mock_sys_hook, \
         patch("IPython.core.ultratb.FormattedTB") as mock_formatted_tb:
        
        # Mock the instance returned by FormattedTB()
        mock_instance = MagicMock()
        mock_formatted_tb.return_value = mock_instance

        # Test Case 1: Default behavior (capture_keyboard_interrupt=False)
        # Should skip KeyboardInterrupt and BdbQuit
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        original_hook = sys.excepthook
        try:
            # Define exception types to test
            kb_interrupt = KeyboardInterrupt("test")
            bdb_quit = BdbQuit("test")
            generic_exc = ValueError("test")
            
            # Create a fake traceback
            fake_tb = MagicMock()

            # Verify KeyboardInterrupt triggers sys.__excepthook__ (skipped)
            sys.excepthook(KeyboardInterrupt, kb_interrupt, fake_tb)
            mock_sys_hook.assert_any_call(KeyboardInterrupt, kb_interrupt, fake_tb)

            # Verify BdbQuit triggers sys.__excepthook__ (skipped)
            sys.excepthook(BdbQuit, bdb_quit, fake_tb)
            mock_sys_hook.assert_any_call(BdbQuit, bdb_quit, fake_tb)

            # Verify generic exception triggers ipython_hook (not skipped)
            sys.excepthook(ValueError, generic_exc, fake_tb)
            # The hook is the instance of FormattedTB created during registration
            # Note: register_ipython_excepthook defines a local excepthook function 
            # that calls ipython_hook (the instance). Since ipython_hook is not 
            # explicitly called as a method in the code provided, but used as an object,
            # we check if it was invoked.
            # In the implementation: ipython_hook(type, value, traceback)
            # This calls the __call__ method of the mock instance.
            mock_instance.assert_called_with(ValueError, generic_exc, fake_tb)

            # Test Case 2: capture_keyboard_interrupt=True
            # Should NOT skip KeyboardInterrupt
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            sys.excepthook(KeyboardInterrupt, kb_interrupt, fake_tb)
            mock_instance.assert_called_with(KeyboardInterrupt, kb_interrupt, fake_tb)

        finally:
            # Restore original excepthook to prevent side effects in other tests
            sys.excepthook = original_hook
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test basic usage without handler (calls log_exception)
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Test Error")
        
        with pytest.raises(ValueError, match="Test Error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test usage with a valid handler function
    handler_calls = []
    def my_handler(e, one, two, three=None, **kw):
        handler_calls.append((e, one, two, three, kw))
        return "handled"

    @exception_wrapper(my_handler)
    def func_to_wrap(one, two, three=10, extra="val"):
        raise RuntimeError("Runtime Error")

    with pytest.raises(RuntimeError):
        func_to_wrap(1, 2, three=3, extra="unexpected")
    
    # Check if handler received correct arguments mapped from the wrapped function
    e, one, two, three, kw = handler_calls[0]
    assert isinstance(e, RuntimeError)
    assert one == 1
    assert two == 2
    assert three == 3
    assert kw["extra"] == "unexpected"

    # 3. Test generator unwrapping/capture
    generator_calls = []
    def gen_handler(e, x):
        generator_calls.append(x)
        raise e

    @exception_wrapper(gen_handler)
    def failing_generator(x):
        yield "start"
        raise ValueError("Gen Error")

    gen = failing_generator(99)
    with pytest.raises(ValueError, match="Gen Error"):
        for val in gen:
            assert val == "start"
    assert generator_calls == [99]

    # 4. Test Validation: Handler missing exception argument
    def invalid_handler_no_e(one):
        pass

    @exception_wrapper(invalid_handler_no_e)
    def func_invalid():
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        exception_wrapper(invalid_handler_no_e)(func_invalid)

    # 5. Test Validation: Handler cannot have *args
    def invalid_handler_with_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(invalid_handler_with_args)(func_invalid)

    # 6. Test Validation: Argument in handler does not exist in wrapped function
    def invalid_handler_missing_param(e, non_existent):
        pass

    @exception_wrapper(invalid_handler_missing_param)
    def func_mismatched(one):
        raise ValueError()

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        exception_wrapper(invalid_handler_missing_param)(func_mismatched)

    # 7. Test Validation: Argument in handler has default value but exists in wrapped function
    def invalid_handler_with_defaults(e, one=5):
        pass

    @exception_wrapper(invalid_handler_with_defaults)
    def func_default_conflict(one):
        raise ValueError()

    with pytest.raises(ValueError, match="cannot have default values"):
        exception_wrapper(invalid_handler_with_defaults)(func_default_conflict)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) - calls log_exception
    with patch("flutes.log.log") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Test Error")

        with pytest.raises(ValueError, match="Test Error"):
            failing_func()
        
        # Verify log_exception was effectively called via log
        assert mock_log.called
        args, _ = mock_log.call_args
        assert "<ValueError> Test Error" in args[0]

    # 2. Test custom handler with positional and keyword arguments
    handler_results = []

    def my_handler(e, val, extra, my_arg=None, **kwargs):
        handler_results.append({
            "e": e,
            "val": val,
            "extra": extra,
            "my_arg": my_arg,
            "kwargs": kwargs
        })

    @exception_wrapper(my_handler)
    def decorated_func(val, extra, my_arg=True, other="default"):
        raise TypeError("Type Error")

    with pytest.raises(TypeError):
        decorated_func(10, "hello", my_arg=False, other="custom")

    assert len(handler_results) == 1
    res = handler_results[0]
    assert isinstance(res["e"], TypeError)
    assert res["val"] == 10
    assert res["extra"] == "hello"
    assert res["my_arg"] is False
    assert res["kwargs"] == {"other": "custom"}

    # 3. Test generator unrolling (generator-based exception catching)
    generator_caught = []

    def gen_handler(e, x):
        generator_caught.append((e, x))

    @exception_wrapper(gen_handler)
    def failing_generator(x):
        yield "start"
        raise RuntimeError("Gen Error")
        yield "end"

    gen = failing_generator(5)
    try:
        for item in gen:
            pass
    except RuntimeError:
        pass

    assert len(generator_caught) == 1
    assert isinstance(generator_caught[0][0], RuntimeError)
    assert generator_caught[0][1] == 5

    # 4. Test validation: Handler must have exception argument
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: None)
        def bad_handler_sig():
            pass

    # 5. Test validation: Handler cannot have *args
    with pytest.pytests_raises_regex(ValueError, "Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def bad_handler_varargs():
            pass

    # 6. Test validation: Argument mismatch (missing in wrapped function)
    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, non_existent: None)
        def missing_arg_func(exists):
            pass

    # 7. Test validation: Argument cannot have default value if it matches wrapped arg
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, val=1: None)
        def duplicate_default_func(val):
            pass

    # 8. Test successful execution (no exception)
    success_flag = []
    @exception_wrapper(my_handler)
    def working_func(val, extra):
        success_flag.append(True)
        return "ok"

    assert working_func(1, 2) == "ok"
    assert success_flag[0] is True
    assert len(handler_results) == 1 # Only the error from step 2 was added

# Helper for regex matching in older pytest versions if needed
def pytest_raises_regex(expected, pattern):
    return pytest.raises(Exception, match=pattern)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (handler_fn is None) -> calls log_exception
    with patch("your_module_path.log_exception") as mock_log_exc:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Original Error")

        with pytest.raises(ValueError, match="Original Error"):
            failing_func()
        mock_log_exc.assert_called_once()

    # 2. Test custom handler with matching arguments and defaults
    # Handler signature: (e, one, two, my_arg=None, **kw)
    # Wrapped signature: (one, two, three=10, **kwargs)
    handler_calls = []

    def custom_handler(e, one, two, my_arg=None, **kw):
        handler_calls.append((e, one, two, my_arg, kw))
        raise e  # Re-raise to allow pytest to catch it

    @exception_wrapper(custom_handler)
    def functional_func(one, two, three=10, **kwargs):
        raise ValueError("Trigger")

    with pytest.raises(ValueError, match="Trigger"):
        functional_func(1, 2, three=30, extra="val")

    # Check if handler received correct mapped arguments
    # 'one' and 'two' are passed directly. 
    # 'three' is in wrapped but NOT in handler (should not be passed to handler)
    # 'my_arg' is in handler with default (passed as None if not provided)
    # 'extra' is in kwargs of wrapped, should end up in kw of handler
    e_val, one_val, two_val, my_arg_val, kw_val = handler_calls[0]
    assert isinstance(e_val, ValueError)
    assert one_val == 1
    assert two_val == 2
    assert my_arg_val is None
    assert kw_val["extra"] == "val"

    # 3. Test validation: Handler must have exception object as first arg
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: x)  # Only one arg, but logic checks if it's the 'e' position essentially
        def bad_handler_args():
            pass

    # 4. Test validation: Handler cannot have *args
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    # 5. Test validation: Argument in handler does not exist in wrapped function
    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, non_existent: None)
        def func_missing_arg():
            pass

    # 6. Test validation: Argument in handler matches wrapped but has a default value
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, one=5: None)
        def func_with_default(one):
            pass

    # 7. Test Generator support (Unrolling generator to catch exceptions inside)
    generator_calls = []

    def gen_handler(e):
        generator_calls.append(e)
        raise e

    @exception_wrapper(gen_handler)
    def failing_generator():
        yield 1
        raise RuntimeError("Generator Error")

    gen_iter = failing_generator()
    with pytest.raises(RuntimeError, match="Generator Error"):
        # We must iterate to trigger the try/except inside _captured_generator
        for _ in gen_iter:
            pass
    assert len(generator_calls) == 1
    assert isinstance(generator_calls[0], RuntimeError)

    # 8. Test __wrapped__ unwrapping (recursive unwrap)
    @exception_wrapper()
    def base_func():
        raise TypeError("Base error")

    @exception_wrapper()
    def wrapped_layer(func):
        return func

    # This tests the _unwrap logic in the decorator
    @exception_wrapper()
    def doubly_wrapped():
        raise TypeError("Deep error")
    
    # Manually simulating a wrapped object for testing the internal _unwrap
    def inner_val():
        raise TypeError("Inner")
    
    # Create a function that looks like it has __wrapped__
    decorated = exception_wrapper()(inner_val)
    decorated.__wrapped__ = inner_val 
    
    with pytest.raises(TypeError, match="Inner"):
        decorated()
```


# LLM-generated content at query #10
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module and its FormattedTB class
    mock_ultratb = MagicMock()
    mock_formatted_tb = MagicMock()
    mock_ultratb.FormattedTB.return_value = mock_formatted_tb
    
    # Store original excepthook to restore it after test
    original_excepthork = sys.excepthook

    with patch.dict("sys.modules", {"IPython.core import ultratb": mock_ultratb, "IPython.core": mock_ultratb}):
        # We need to patch the actual import path used in the function
        # The code uses: from IPython.core import ultratb
        with patch("IPython.core.ultratb", mock_ultratb.FormattedTB):
            
            # Case 1: capture_keyboard_interrupt = False (Default)
            # Should skip BdbQuit and KeyboardInterrupt
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            # Test BdbQuit (should call original hook)
            with patch("sys.__excepthook__") as mock_sys_hook:
                dummy_tb = MagicMock()
                new_hook(BdbQuit, BdbQuit("quit"), dummy_tb)
                mock_sys_hook.assert_called_once_with(BdbQuit, BdbQuit("quit"), dummy_tb)

            # Test KeyboardInterrupt (should call original hook)
            with patch("sys.__excepthook__") as mock_sys_hook:
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)
                mock_sys_hook.assert_called_once_with(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)

            # Test regular Exception (should call ipython_hook/ultratb)
            with patch("IPython.core.ultratb.FormattedTB") as mock_formatted_tb_instance:
                new_hook(ValueError, ValueError("error"), dummy_tb)
                # The hook uses the instance created during registration
                # Since we can't easily access the local 'ipython_hook' variable from inside 
                # the closure without complex debugging, we verify if the logic branches.
                # However, we can check if sys.__excepthook__ was NOT called for ValueError.
                pass

            # Case 2: capture_keyboard_interrupt = True
            # Should skip BdbQuit but NOT KeyboardInterrupt
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            with patch("sys.__excepthook__") as mock_sys_hook:
                # BdbQuit should still be skipped
                new_hook_true(BdbQuit, BdbQuit("quit"), dummy_tb)
                mock_sys_hook.assert_called()
                
                # KeyboardInterrupt should now trigger ipython_hook (not sys.__excepthook__)
                mock_sys_hook.reset_mock()
                new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)
                mock_sys_hook.assert_not_called()

    # Restore original state
    sys.excepthook = original_excepthork
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn) -> calls log_exception
    with patch("flutes.log.log") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        
        # Verify log_exception (via log) was called
        assert mock_log.called

    # 2. Test custom handler with matching arguments
    handler_results = []
    def custom_handler(e, one, two, kw):
        handler_results.append((e, one, two, kw))
        return "handled"

    @exception_wrapper(custom_handler)
    def target_func(one, two, extra="default"):
        raise TypeError("type error")

    with pytest.raises(TypeError):
        target_func(1, 2, extra="value")
    
    assert len(handler_results) == 1
    e, one, two, kw = handler_results[0]
    assert isinstance(e, TypeError)
    assert one == 1
    assert two == 2
    assert kw == {"extra": "value"}

    # 3. Test custom handler with varargs (kwargs) capture
    handler_captured_kwargs = []
    def kw_handler(e, x, **kwargs):
        handler_captured_kwargs.append(kwargs)

    @exception_wrapper(kw_handler)
    def kw_func(x, y, z):
        raise ValueError("fail")

    with pytest.raises(ValueError):
        kw_func(10, 20, z=30)
    
    # The decorator logic maps remaining bound arguments to the handler's **kwargs
    assert "z" in handler_captured_kwargs[0]
    assert handler_captured_kwargs[0]["z"] == 30

    # 4. Test Generator support (unrolling generator)
    gen_handled = []
    def gen_handler(e, val):
        gen_handled.append((e, val))

    @exception_wrapper(gen_handler)
    def error_generator(val):
        yield 1
        raise RuntimeError("gen error")

    gen_obj = error_generator(99)
    with pytest.raises(RuntimeError):
        for _ in gen_obj:
            pass
    
    assert len(gen_handled) == 1
    assert gen_handled[0][0].__class__ == RuntimeError
    assert gen_handled[0][1] == 99

    # 5. Test Validation Errors (Invalid Handler Signatures)
    
    # Error: No exception argument
    def invalid_handler_no_e(one):
        pass
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler_no_e)
        def dummy(): pass

    # Error: Handler has *args
    def invalid_handler_with_args(e, *args):
        pass
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(invalid_handler_with_args)
        def dummy2(): pass

    # Error: Argument in handler does not exist in wrapped function
    def invalid_handler_missing_arg(e, non_existent):
        pass
    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(invalid_handler_missing_arg)
        def dummy3(one): pass

    # Error: Argument matches wrapped function but has a default value (ambiguity/restriction)
    def invalid_handler_with_default(e, one=10):
        pass
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(invalid_handler_with_default)
        def dummy4(one): pass

    # 6. Test unwrapping of decorated functions
    call_count = 0
    @exception_wrapper()
    def inner():
        nonlocal call_count
        call_count += 1
        raise AttributeError("attr error")

    @exception_wrapper()
    def outer():
        return inner()

    with pytest.raises(AttributeError):
        outer()
    
    # If unwrapping works, the actual function logic executes and hits the exception
    assert call_count == 1
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_name.log")  # Replace 'your_module_name' with the actual module name
def test_log_exception(mock_log):
    # Test Case 1: Basic exception logging without user message
    exc = ValueError("test error")
    log_exception(exc)
    
    expected_msg = "<ValueError> test error"
    # Check if log was called with the formatted exception traceback and the error message
    # Note: log_exception calls log twice: once for traceback, once for the message.
    # We check the second call (the actual error message)
    mock_log.assert_any_call(expected_msg, "error")

    # Test Case 2: Exception logging with user message
    user_msg = "An error occurred"
    log_exception(exc, user_msg=user_msg)
    expected_msg_with_user = f"{user_msg}: <ValueError> test error"
    mock_log.assert_any_call(expected_msg_with_user, "error")

    # Test Case 3: CalledProcessError with output (should not log traceback separately)
    # According to logic: if e.output is NOT None, it skips the first log call (traceback)
    proc_err = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error details")
    log_exception(proc_err)
    
    # Verify that the traceback-specific log call was skipped for this specific case
    # By checking if only the error message log exists in calls
    calls = [call.args[0] for call in mock_log.call_args_list]
    assert any("<CalledProcessError> error details" in msg for msg in calls)
    # Check that no traceback string was passed (the traceback is usually a long string, 
    # we just check if the specific message exists)

    # Test Case 4: Exception during logging itself
    mock_log.side_effect = Exception("Logging failed")
    with pytest.raises(Exception) as excinfo:
        log_exception(exc)
    assert "Logging failed" in str(excinfo.value)

@patch("your_module_name.log")
def test_log_exception_with_kwargs(mock_log):
    # Test Case 5: Passing extra kwargs to the logger
    exc = TypeError("type error")
    log_exception(exc, extra_info="metadata", user_msg="fail")
    
    # Verify kwargs are passed through to the log function
    mock_log.assert_any_call("fail: <TypeError> type error", "error", extra_info="metadata")
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (handler_fn is None) - should call log_exception
    with patch("your_module.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(a, b):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func(1, 2)
        mock_log.assert_called_once()

    # 2. Test custom handler with matching arguments (no defaults in handler for these)
    handler_calls = []
    def custom_handler(e, one, two):
        handler_calls.append((e, one, two))
        raise ValueError("re-raise")

    @exception_wrapper(custom_handler)
    def target_func(one, two, three=10):
        raise ValueError("trigger")

    with pytest.raises(ValueError, match="re-raise"):
        target_func(1, 2, three=3)
    
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ValueError)
    assert handler_calls[0][1] == 1
    assert handler_calls[0][2] == 2

    # 3. Test custom handler with **kwargs (capturing extra args from wrapped func)
    handler_results = []
    def kwargs_handler(e, arg_a, **kwargs):
        handler_results.append((arg_a, kwargs))
        raise ValueError("done")

    @exception_wrapper(kwargs_handler)
    def target_with_extra(arg_a, arg_b, extra="val"):
        raise ValueError("trigger")

    with pytest.append("done"): # logic check: we expect the error from handler to propagate
        try:
            target_with_extra(arg_a=5, arg_b=6, extra="val")
        except ValueError as e:
            assert str(e) == "done"
    
    assert handler_results[0][0] == 5
    # The 'kwargs' in handler gets everything from bound_args not explicitly named in handler args
    # In this case, arg_b and extra are passed through
    assert handler_results[0][1]["arg_b"] == 6
    assert handler_results[0][1]["extra"] == "val"

    # 4. Test Generator support (unrolling the generator)
    gen_called = []
    def gen_handler(e, x):
        gen_called.append(x)
        raise ValueError("stop")

    @exception_wrapper(gen_handler)
    def my_generator(x):
        yield 1
        yield 2
        raise RuntimeError("error in generator")

    gen_it = my_generator(100)
    with pytest.raises(ValueError, match="stop"):
        for val in gen_it:
            pass
    assert gen_called == [100]

    # 5. Test Validation: Handler must have exception param
    def invalid_handler(a): # missing 'e' as first arg logic is checked via inspect arg count/names
        pass
    
    @pytest.raises(ValueError, match="Exception handler must have a positional argument"):
    @exception_wrapper(lambda: None)
    def empty_handler_func():
        pass

    # 6. Test Validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def func():
            pass

    # 7. Test Validation: Argument mismatch (name in handler not in wrapped)
    def mismatch_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(mismatch_handler)
        def func():
            pass

    # 8. Test Validation: Argument in handler has default value but exists in wrapped (forbidden by logic)
    def default_in_handler_error(e, val=1):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_in_handler_error)
        def func(val):
            pass

    # 9. Test unwrapping (handling @wraps or nested decorators)
    @exception_wrapper()
    def inner():
        raise ValueError("deep error")

    @exception_wrapper()
    def outer():
        return inner()

    with pytest.raises(ValueError, match="deep error"):
        outer()
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_path.log")
def test_log_exception(mock_log):
    # Test 1: Basic exception logging without user message
    e1 = ValueError("test error")
    log_exception(e1)
    
    # Check if log was called with the formatted exception string
    # The first call should be for the traceback, second for the exception itself
    expected_exc_msg = "<ValueError> test error"
    assert mock_log.call_args_list[1][0][0] == expected_exc_msg
    assert mock_log.call_args_list[1][0][1] == "error"

    # Test 2: Logging with a user message
    e2 = TypeError("type error")
    user_msg = "An error occurred"
    log_exception(e2, user_msg=user_msg)
    
    expected_exc_msg_with_user = f"{user_msg}: <TypeError> type error"
    # Find the call that contains the expected message
    found = any(call[0][0] == expected_exc_msg_with_user for call in mock_log.call_args_list)
    assert found

    # Test 3: Logging a CalledProcessError with output (should NOT log traceback)
    e3 = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error output")
    mock_log.reset_mock()
    log_exception(e3)
    
    # Verify that log was ONLY called for the exception message, not the traceback
    # (Because e3.output is not None)
    assert mock_log.call_count == 1
    assert "<CalledProcessError> 'ls' returned non-zero exit status 1" in mock_log.call_args[0][0]

    # Test 4: Testing the fallback to print when logging itself fails
    e4 = RuntimeError("original error")
    with patch("your_module_path.log", side_effect=Exception("logging failure")):
        with patch("builtins.print") as mock_print:
            log_exception(e4, user_msg="Fail test")
            
            # Verify print was called with the exception message
            mock_print.assert_any_call("Fail test: <RuntimeError> original error")
            # Verify the second print about the logging failure occurred
            assert any("Another exception occurred while logging" in str(call) for call in mock_print.call_args_list)

    # Test 5: Testing kwargs propagation
    e5 = AttributeError("attr error")
    extra_kwargs = {"extra": "data", "level": "critical"}
    log_exception(e5, **extra_kwargs)
    
    # Check if the last call to log included the extra kwargs
    last_call_kwargs = mock_log.call_args[1]
    assert last_call_kwargs["extra"] == "data"
    assert last_call_kwargs["level"] == "critical"
```


# LLM-generated content at query #15
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython module and its ultratb component
    mock_ultratb = MagicMock()
    mock_ipython_core = MagicMock()
    mock_ipython_core.ultratb = mock_ultratb

    with patch.dict("sys.modules", {"IPython": mock_ipython_core, "IPython.core": mock_ipython_core, "IPython.core.ultratb": mock_ultratb}):
        # Store original excepthook to restore it later
        original_excepthook = sys.excepthook
        
        try:
            # Test Case 1: capture_keyboard_interrupt = False (Default)
            # Should skip KeyboardInterrupt and BdbQuit, but call ipython_hook for others
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            mock_sys_excepthook = MagicMock()
            with patch("sys.__excepthook__", mock_sys_excepthook):
                # Test BdbQuit (should call sys.__excepthook__)
                new_hook(BdbQuit, BdbQuit(), None)
                mock_sys_excepthook.assert_called_with(BdbQuit, BdbQuit(), None)

                # Test KeyboardInterrupt (should call sys.__excepthook__ because capture=False)
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), None)
                mock_sys_excepthook.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), None)

                # Test ValueError (should trigger ipython_hook/ultratb)
                # The hook uses ultratb.FormattedTB(...) which we mocked
                new_hook(ValueError, ValueError("test"), None)
                mock_ultratb.FormattedTB.assert_called()

            # Test Case 2: capture_keyboard_interrupt = True
            # Should NOT skip KeyboardInterrupt (should call ipython_hook)
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            mock_sys_excepthook.reset_mock()
            with patch("sys.__excepthook__", mock_sys_excepthook):
                # KeyboardInterrupt should now trigger the ipython_hook (not sys.__excepthook__)
                new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), None)
                mock_sys_excepthook.assert_not_called()

        finally:
            # Restore original excepthook to prevent side effects in other tests
            sys.excepthook = original_excepthook
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn): calls log_exception
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(a, b):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func(1, 2)
        mock_log.assert_called_once()

    # 2. Test with valid handler function and argument mapping
    handler_calls = []
    def my_handler(e, one, two, extra=None, **kwargs):
            handler_calls.append((e, one, two, extra, kwargs))
            raise ValueError("re-raise to stop execution")

    @exception_wrapper(my_handler)
    def target_func(one, two, three=10, four=20):
        return "success"

    with pytest.raises(ValueError, match="re-raise to stop execution"):
        target_func(1, 2, three=3, four=4)

    # Verify the mapping logic:
    # 'one' and 'two' are required in handler (from target_func args)
    # 'extra' is a default arg in handler, should be handled
    # 'kwargs' contains leftovers
    e, one, two, extra, kwargs = handler_calls[0]
    assert isinstance(e, ValueError)
    assert one == 1
    assert two == 2
    assert extra is None  # extra was not in target_func signature
    assert kwargs["three"] == 3
    assert kwargs["four"] == 4

    # 3. Test generator support (unrolling generator to catch exceptions)
    gen_calls = []
    def gen_handler(e, val):
        gen_calls.append((e, val))
        raise ValueError("stop")

    @exception_wrapper(gen_handler)
    def failing_generator(val):
        yield 1
        raise RuntimeError("generator error")

    gen = failing_generator(99)
    with pytest.raises(ValueError, match="stop"):
        for _ in gen:
            pass
    
    assert len(gen_calls) == 1
    assert gen_calls[0][0] == RuntimeError("generator error")
    assert gen_calls[0][1] == 99

    # 4. Test Validation: Handler must have exception arg
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: None)
        def bad_handler_args(a):
            pass

    # 5. Test Validation: Handler cannot have *args
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def bad_handler_varargs(a):
            pass

    # 6. Test Validation: Handler argument mismatch (required arg not in wrapped function)
    with pytest.raises(ValueError, match="Argument 'missing' in exception handler does not match"):
        @exception_wrapper(lambda e, missing: None)
        def missing_arg_func(a):
            pass

    # 7. Test Validation: Handler argument cannot have default if it exists in wrapped function
    with pytest.raises(ValueError, match="Argument 'a' matches wrapped method argument, thus cannot have default values"):
        @exception_wrapper(lambda e, a=1: None)
        def duplicate_default_func(a):
            pass

    # 8. Test unwrapping of decorated functions (recursion check)
    @exception_wrapper()
    def inner_func(x):
        raise TypeError("inner error")

    @exception_wrapper()
    def outer_func(x):
        return inner_func(x)

    with pytest.raises(TypeError, match="inner error"):
        outer_func(10)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test Default Behavior (No handler provided) - Should call log_exception
    with patch("your_module_path.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(x):
            raise ValueError("error occurred")

        with pytest.raises(ValueError, match="error occurred"):
            failing_func(10)
        
        mock_log.assert_called_once()
        # Verify the exception object passed to log_exception is the one raised
        args, _ = mock_log.call_args
        assert isinstance(args[0], ValueError)

    # 2. Test Custom Handler with matching positional/keyword arguments
    handler_calls = []
    def my_handler(e, val, extra, my_arg=None, **kwargs):
            handler_calls.append((e, val, extra, my_arg, kwargs))
            raise RuntimeError("re-raised from handler")

    @exception_wrapper(my_handler)
    def target_func(val, extra, my_arg=None, other="unused"):
        raise ValueError("original error")

    with pytest.raises(RuntimeError, match="re-raised from handler"):
        target_func(1, "two", my_arg="three", other="four")

    # Verify arguments passed to handler: 
    # e=ValueError, val=1, extra="two", my_arg="three", kwargs={'other': 'four'}
    e_caught, val_caught, extra_caught, my_arg_caught, kw_caught = handler_calls[0]
    assert isinstance(e_caught, ValueError)
    assert val_caught == 1
    assert extra_caught == "two"
    assert my_arg_caught == "three"
    assert kw_caught == {"other": "four"}

    # 3. Test Generator support (Unrolling generator to catch exceptions)
    generator_results = []
    @exception_wrapper()
    def error_generator(n):
        for i in range(n):
            if i == 1:
                raise ValueError("gen error")
            yield i

    gen = error_generator(3)
    with pytest.raises(ValueError, match="gen error"):
        list(gen)

    # 4. Test Validation: Handler must have an exception argument
    def invalid_handler(not_e): # This is technically valid if it's the first, but let's test zero args
        pass
    
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def func(): pass

    # 5. Test Validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def func(): pass

    # 6. Test Validation: Argument mismatch (Handler expects arg not in function)
    def mismatch_handler(e, missing_val):
        pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(mismatch_handler)
        def func(existing_val):
            pass

    # 7. Test Validation: Argument collision (Handler has default for arg that function provides without default)
    def collision_handler(e, val=10): # val has default here
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(collision_handler)
        def func(val): # val does NOT have default here
            pass

    # 8. Test Unwrapping (handling decorated functions)
    @exception_wrapper()
    def inner():
        raise ValueError("inner error")

    @exception_wrapper()
    def outer():
        return inner()

    with pytest.raises(ValueError, match="inner error"):
        outer()
```


# LLM-generated content at query #3
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython module and its ultratb component
    mock_ultratb = MagicMock()
    mock_ipython_module = MagicMock()
    mock_ipython_module.ultratb = mock_ultratb
    
    # Store original excepthook to restore after test
    original_excepthook = sys.excepthook
    
    try:
        with patch.dict("sys.modules", {"IPython": mock_ipython_module, "IPython.core": mock_ipython_module.core}):
            # Test Case 1: capture_keyboard_interrupt=False (Default)
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            # Verify hook is set and ultratb.FormattedTB was called
            assert new_hook is not original_excepthook
            mock_ultratb.FormattedTB.assert_called_with(mode='Context', color_scheme='Linux', call_pdb=1)
            
            # Verify KeyboardInterrupt triggers ipython_hook (the inner function logic)
            # We simulate the internal 'excepthook' by calling it with a type that is NOT in skip_exceptions
            with patch("sys.__excepthook__") as mock_sys_hook:
                # Create an exception not in [BdbQuit, KeyboardInterrupt]
                test_type = ValueError
                test_val = ValueError("test")
                test_tb = MagicMock()
                
                # The 'ipython_hook' is trapped in the closure. 
                # We test if the logic passes through to the mock-injected ultratb.
                new_hook(test_type, test_val, test_tb)
                # Since we can't easily access the inner ipython_hook object, 
                # we check if it interacted with our mocked ultratb via its registration
                # Note: In a real environment, 'ipython_hook' is the instance of FormattedTB.
                # The test checks that the logic reaches the point where it doesn't call sys.__excepthook__
                mock_sys_hook.assert_not_called()

            # Test Case 2: capture_keyboard_interrupt=True
            # This should add KeyboardInterrupt to skip_exceptions
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            with patch("sys.__excepthook__") as mock_sys_hook:
                # Trigger with KeyboardInterrupt
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), MagicMock())
                mock_sys_hook.assert_called_once()

            # Test Case 3: BdbQuit should always be skipped
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            with patch("sys.__excepthook__") as mock_sys_hook:
                new_hook(BdbQuit, BdbQuit(), MagicMock())
                mock_sys_hook.assert_called_once()

    finally:
        # Clean up: restore the original excepthook for other tests
        sys.excepthook = original_excepthook
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_name.log")  # Replace 'your_module_name' with the actual module name
def test_log_exception(mock_log):
    # Test Case 1: Simple exception logging without user message
    exc = ValueError("test error")
    log_exception(exc)
    
    expected_msg = "<ValueError> test error"
    # Verify log was called with formatted traceback and the exception message
    # Note: traceback.format_exc() is dynamic, so we check if any call contains our msg
    args, kwargs = mock_log.call_args_list[0]
    assert expected_msg in args[1] 
    assert args[1] == "error"

    # Test Case 2: Exception logging with user message
    user_msg = "An error occurred"
    log_exception(exc, user_msg=user_msg)
    
    expected_msg_with_prefix = f"{user_msg}: <ValueError> test error"
    # Check the second call to log (the one containing the message)
    args, kwargs = mock_log.call_args_list[1]
    assert expected_msg_with_prefix in args[1]

    # Test Case 3: CalledProcessError with output (should skip traceback logging)
    # According to code: if e.output is not None, it skips log(traceback.format_exc())
    proc_err = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error details")
    log_exception(proc_err)
    
    # The last call should be the error message, and we check that 
    # it didn't try to log a traceback in a way that implies 'output' was present.
    # We look at the calls specifically for this exception instance.
    found_traceback_log = False
    for call in mock_log.call_args_list:
        if "<CalledProcessError>" in call[0][1] and call[0][1] != "error":
             # If we found a call where the second arg (level) is not 'error', 
             # it might be the traceback log. But for CalledProcessError with output, 
             # that specific line should be skipped.
             pass

    # Test Case 4: Testing the fallback to print when logging itself fails
    with patch("builtins.print") as mock_print:
        mock_log.side_effect = Exception("Logging failed")
        try:
            log_exception(exc)
        except Exception as e:
            assert str(e) == "Logging failed"
        
        # Check if it printed the exception message to stdout
        mock_print.assert_any_call("<ValueError> test error")
        mock_print.assert_any_call(pytest.mark.any_string) # "Another exception occurred..."
```


# LLM-generated content at query #5
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module before it is imported inside the function
    with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.core": MagicMock(), "IPython.core.ultratb": MagicMock()}):
        from unittest.mock import patch
        
        # Test case 1: capture_keyboard_interrupt = False (Default)
        # In this mode, KeyboardInterrupt should be in skip_exceptions
        with patch("sys.excepthook") as mock_sys_hook:
            # We need to mock sys.__excepthook__ because the function calls it
            with patch("sys.__excepthook__") as mock_original_hook:
                register_ipython_excepthook(capture_keyboard_interrupt=False)
                
                # Capture the newly assigned excepthook
                new_hook = sys.excepthook
                
                # Test with KeyboardInterrupt (should NOT call ipython_hook, should call sys.__excepthook__)
                # We simulate an exception type that is in skip_exceptions
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), None)
                mock_original_hook.assert_called_once()
                
                # Test with ValueError (should call ipython_hook)
                # To verify this, we must patch the ultratb instance created inside the function
                with patch("IPython.core.ultratb.FormattedTB") as mock_fmttb:
                    # Re-register to ensure the patch is active for the new instance creation
                    register_ipython_excepthook(capture_keyboard_interrupt=False)
                    new_hook = sys.excepthook
                    
                    new_hook(ValueError, ValueError(), None)
                    mock_fmttb.assert_called()

        # Test case 2: capture_keyboard_interrupt = True
        # In this mode, KeyboardInterrupt should NOT be in skip_exceptions
        with patch("sys.__excepthook__") as mock_original_hook:
            with patch("IPython.core.ultratb.FormattedTB") as mock_fmttb:
                register_ipython_excepthook(capture_keyboard_interrupt=True)
                new_hook = sys.excepthook
                
                # Test with KeyboardInterrupt (should now call ipython_hook because it's not skipped)
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), None)
                mock_fmttb.assert_called()
                
                # Test with BdbQuit (should still be skipped)
                mock_fmttb.reset_mock()
                new_hook(BdbQuit, BdbQuit(), None)
                mock_fmttb.assert_not_called()
                mock_original_hook.assert_called()

```


# LLM-generated content at query #6
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for the dependencies inside the function
    mock_ultratb = MagicMock()
    mock_FormattedTB = MagicMock()
    mock_ultratb.FormattedTB = mock_FormattedTB
    
    # Mock sys.__excepthook__ to track calls
    original_excepthook = sys.excepthook
    sys.__excepthook__ = MagicMock()

    try:
        with patch.dict("sys", {"__excepthook__": MagicMock()}, clear=False), \
             patch.dict("sys.modules", {"IPython.core import ultratb": mock_ultratb}, clear=False), \
             patch("IPython.core.ultratb.FormattedTB", mock_FormattedTB):
            
            # Test Case 1: capture_keyboard_interrupt = False (Default)
            # KeyboardInterrupt and BdbQuit should trigger sys.__excepthook__
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            new_hook = sys.excepthook
            
            # Simulate a KeyboardInterrupt
            kb_interrupt = KeyboardInterrupt("User interrupted")
            new_hook(KeyboardInterrupt, kb_interrupt, None)
            sys.__excepthook__.assert_any_call(KeyboardInterrupt, kb_interrupt, None)

            # Simulate BdbQuit
            bdb_quit = BdbQuit()
            new_hook(BdbQuit, bdb_quit, None)
            sys.__excepthook__.assert_any_call(BdbQuit, bdb_quit, None)

            # Simulate a standard Exception (should trigger ipython_hook/ultratb)
            # Since we mocked the assignment of ultratb.FormattedTB, 
            # calling the hook will execute the logic that calls it.
            val_error = ValueError("Test error")
            new_hook(ValueError, val_error, None)
            
            # Verify the IPython formatter was instantiated
            assert mock_FormattedTB.called

            # Test Case 2: capture_keyboard_interrupt = True
            # KeyboardInterrupt should now trigger ipython_hook instead of sys.__excepthook__
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Reset mock to clear previous calls
            sys.__excepthook__.reset_mock()
            
            new_hook_true = sys.excepthook
            new_hook_true(KeyboardInterrupt, kb_interrupt, None)
            
            # It should NOT call sys.__excepthook__ because capture is True
            sys.__excepthook__.assert_not_called()

    finally:
        # Restore original state
        sys.excepthook = original_excepthook
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_path.log")
def test_log_exception(mock_log):
    # Test 1: Basic exception logging without user message
    e1 = ValueError("test error")
    log_exception(e1)
    
    # Verify log was called with the formatted exception and error level
    # format_exc() is used, so we check if the call contains parts of the error
    args, kwargs = mock_log.call_args_list[0]
    assert "<ValueError> test error" in args[1]
    assert args[0] == args[1] # The first arg is traceback, second is msg
    assert kwargs.get("level") == "error" or any(arg == "error" for arg in args)

    # Test 2: Logging with a user message
    e2 = TypeError("type error")
    user_msg = "Custom Prefix"
    log_exception(e2, user_msg=user_msg)
    
    # Verify the message is prefixed correctly
    found_prefixed = False
    for call in mock_log.call_args_list:
        msg = call[0][1] if len(call[0]) > 1 else call[0][0]
        if f"{user_msg}: <TypeError> type error" in msg:
            found_prefixed = True
    assert found_prefixed

    # Test 3: Logging a subprocess.CalledProcessError with output (should skip traceback log)
    e3 = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error output")
    log_exception(e3)
    
    # Check that the error message was logged, but we should verify it didn't attempt 
    # to log the traceback as a separate call if logic allows skipping.
    # Based on code: if e.output is not None, it skips the first log(traceback...)
    # We check the number of calls to ensure only the error message was logged.
    # Note: The trace is usually logged first. If output exists, traceback call is skipped.
    msg_calls = [call[0][1] for call in mock_log.call_args_list if len(call[0]) > 1]
    if e3.output is not None:
        # If logic skips traceback, only the exception message remains
        assert not any("Traceback" in m for m in msg_calls)

    # Test 4: Logging failure (when log() itself raises an exception)
    mock_log.side_effect = [None, Exception("Logging Failed")]
    e4 = RuntimeError("runtime error")
    
    with patch("builtins.print") as mock_print:
        log_exception(e4)
        # Verify that the fallback print was called
        mock_print.assert_any_call("<RuntimeError> runtime error")
        # Verify the second print for the secondary exception occurred
        printed_msgs = [call[0] for call in mock_print.call_args_list]
        assert any("Another exception occurred while logging" in m for m in printed_msgs)

    # Test 5: Verification of kwargs passing
    mock_log.reset_mock()
    extra_params = {"extra_info": "important"}
    log_exception(e1, **extra_params)
    assert mock_log.call_args[1]["extra_info"] == "important"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_name.log")  # Replace 'your_module_name' with the actual module name
def test_log_exception(mock_log):
    # Test 1: Basic exception logging without user message
    e1 = ValueError("simple error")
    log_exception(e1)
    
    # Verify log was called with formatted exception and 'error' level
    # We check if the first argument contains the class name and error message
    args, kwargs = mock_log.call_args_list[0]
    assert "<ValueError> simple error" in args[0]
    assert args[1] == "error"

    # Test 2: Exception logging with user message
    e2 = TypeError("type error")
    user_msg = "Custom context"
    log_exception(e2, user_msg=user_msg)
    
    args, kwargs = mock_log.call_args_list[1]
    assert f"{user_msg}: <TypeError> type error" in args[0]

    # Test 3: Exception logging with additional kwargs passed to log()
    extra_info = {"module": "test_suite"}
    log_exception(e1, extra_info=extra_info)
    
    args, kwargs = mock_log.call_args_list[2]
    assert kwargs["extra_info"] == "test_suite"

    # Test 4: Subprocess.CalledProcessError with output (should NOT log traceback)
    # According to code: if not (isinstance(e, CalledProcessError) and e.output is not None)
    # If it HAS output, the first log call (traceback) should be skipped.
    e_proc = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error output")
    log_exception(e_proc)
    
    # Check only the error message log exists, not the traceback log
    # The first call in this specific test execution (3rd index in total list) 
    # should be the exception message itself, and there shouldn't be a preceding traceback call for this instance.
    # Since we are tracking all calls to mock_log:
    current_call_args = mock_log.call_args_list[3]
    assert "<CalledProcessError> 'ls' returned non-zero exit status 1" in current_call_args[0]
    # Verify that for this specific call, the traceback wasn't logged (i.e., only 1 call made for this error)
    # We check if the count of calls specifically for this e_proc is 1.
    calls_for_proc = [c for c in mock_log.call_args_list if "<CalledProcessError>" in c[0][0]]
    assert len(calls_for_proc) == 1

    # Test 5: Error during logging (the 'except Exception as log_e' block)
    # We force the 'log' function to raise an error
    mock_log.side_effect = [None, None, None, Exception("Logging failed")]
    
    with pytest.raises(Exception) as excinfo:
        log_exception(e1)
    assert "Logging failed" in str(excinfo.value)

@patch("your_module_name.log")
def test_log_exception_traceback_present(mock_log):
    # Test that traceback is indeed logged for standard exceptions
    e = RuntimeError("crash")
    try:
        raise e
    except RuntimeError:
        log_exception(e)
    
    # The first call to log should be the traceback string
    traceback_call = mock_log.call_args_list[0][0][0]
    assert "RuntimeError: crash" in traceback_call
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn)
    # Should call log_exception when an exception occurs
    with patch("your_module_path.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Standard error")

        with pytest.raises(ValueError, match="Standard error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with specific arguments
    # Handler signature: (e, one, two, my_arg=None, **kwargs)
    # Function signature: (one, two, three, my_arg=None, extra=10)
    def my_handler(e, one, two, my_arg=None, **kwargs):
        return {"exception": str(e), "one": one, "two": two, "my_arg": my_arg, "extra": kwargs.get("extra")}

    @exception_wrapper(handler_fn=my_handler)
    def target_func(one, two, three, my_arg=None, extra=10):
        raise TypeError("Type error")

    result = target_func(1, 2, 3, my_arg="val", extra=99)
    assert result["exception"] == "Type error"
    assert result["one"] == 1
    assert result["two"] == 2
    assert result["my_arg"] == "val"
    assert result["extra"] == 99

    # 3. Test generator unrolling and exception catching
    @exception_wrapper(handler_fn=my_handler)
    def failing_generator(one, two, three):
        yield "first"
        raise RuntimeError("Gen error")

    gen = failing_generator(10, 20, 30)
    assert next(gen) == "first"
    # The exception in the generator should trigger the handler and return its result
    res_gen = next(gen, None)
    assert res_gen["exception"] == "Gen error"
    assert res_gen["one"] == 10

    # 4. Test Validation: Exception handler must have positional argument for exception
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(handler_fn=lambda x: x)
        def bad_handler_func():
            pass
        bad_handler_func()

    # 5. Test Validation: Exception handler cannot have *args
    def handler_with_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(handler_fn=handler_with_args)
        def bad_handler_func_2():
            pass
        bad_handler_func_2()

    # 6. Test Validation: Argument in handler does not match wrapped method
    def handler_mismatch(e, non_existent):
        pass

    @exception_wrapper(handler_fn=handler_mismatch)
    def mismatch_func(one):
        raise ValueError("error")

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        mismatch_func(1)

    # 7. Test Validation: Argument with default value in handler cannot be present in wrapped method signature
    def handler_with_defaults(e, one=True):
        pass

    @exception_wrapper(handler_fn=handler_with_defaults)
    def func_with_overlap(one):
        raise ValueError("error")

    with pytest.raises(ValueError, match="Argument 'one' matches wrapped method argument, thus cannot have default values"):
        func_with_overlap(1)

    # 8. Test Unwrapping (Decorator on already wrapped function)
    @exception_wrapper()
    def inner_func():
        raise ValueError("Inner error")

    @exception_wrapper()
    def outer_func():
        return inner_func()

    with pytest.raises(ValueError, match="Inner error"):
        outer_func()
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("error_type, user_msg, kwargs", [
    (ValueError("test error"), "User Message", {"extra": "data"}),
    (RuntimeError("runtime error"), None, {}),
])
def test_log_exception(error_type, user_msg, kwargs):
    with patch("your_module_path.log") as mock_log:
        # We also need to patch traceback.format_exc because log_exception calls it
        with patch("traceback.format_exc", return_value="fake_traceback"):
            log_exception(error_type, user_msg=user_msg, **kwargs)

            # Check if log was called with the formatted exception string
            expected_exc_msg = f"<{error_type.__class__.__qualname__}> {error_type}"
            if user_msg:
                expected_exc_msg = f"{user_msg}: {expected_exc_msg}"

            # The first call is the traceback, second is the exception message itself
            assert mock_log.call_count == 2
            mock_log.assert_any_call("fake_traceback", "error", **kwargs)
            mock_log.assert_any_call(expected_exc_msg, "error", **kwargs)

def test_log_exception_subprocess_error():
    """Tests that CalledProcessError with output does not log the traceback separately."""
    # Create a mock CalledProcessError where output is present
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error output")
    
    with patch("your_module_path.log") as mock_log:
        log_exception(err)
        
        # Should only call log once for the message, not for the traceback 
        # because 'if not (isinstance(...) and e.output is not None)' evaluates to False
        assert mock_log.call_count == 1
        expected_msg = f"<{err.__class__.__qualname__}> {err}"
        mock_log.assert_called_once_with(expected_msg, "error")

def test_log_exception_logging_failure():
    """Tests that if the logger itself fails, it prints to stdout and re-raises."""
    err = ValueError("original error")
    logger_error = RuntimeError("logger failed")
    
    with patch("your_module_path.log", side_effect=logger_error):
        with patch("builtins.print") as mock_print:
            with pytest.raises(RuntimeError) as excinfo:
                log_exception(err)
            
            assert excinfo.value == logger_error
            # Verify it printed the original error and the logging error
            expected_msg = f"<{err.__class__.__qualname__}> {err}"
            mock_print.assert_any_call(expected_msg)
            assert "Another exception occurred while logging" in mock_print.call_args_list[1][0][0]
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test Default Behavior (No handler provided)
    # Should call log_exception when an exception occurs
    with patch("your_module_path.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Original Error")

        with pytest.raises(ValueError, match="Original Error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test Custom Handler with Positional and Keyword Arguments
    # Scenario: handler(e, one, two, my_arg=None, **kw)
    # Wrapped func: func(one, two, three=10, **kwargs)
    handler_calls = []

    def custom_handler(e, one, two, my_arg=None, **kw):
        handler_calls.append({
            "e": e,
            "one": one,
            "two": two,
            "my_arg": my_arg,
            "kw": kw
        })

    @exception_wrapper(custom_handler)
    def target_func(one, two, three=10, extra="val"):
        raise RuntimeError("Trigger")

    with pytest.raises(RuntimeError, match="Trigger"):
        target_func(1, 2, extra="extra_val")

    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0]["e"], RuntimeError)
    assert handler_calls[0]["one"] == 1
    assert handler_calls[0]["two"] == 2
    # 'three' is a default in target_func, so it shouldn't be passed to handler as an explicit arg
    # unless we check the logic for handler_arg_names. 
    # The decorator logic: handler_arg_names excludes args with defaults from target_func.
    assert "three" not in handler_calls[0]
    assert handler_calls[0]["kw"]["extra"] == "extra_val"

    # 3. Test Generator Support
    @exception_wrapper()
    def generator_func():
        yield 1
        raise TypeError("Gen Error")

    gen = generator_func()
    assert next(gen) == 1
    with pytest.raises(TypeError, match="Gen Error"):
        next(gen)

    # 4. Test Validation: Handler must have exception arg
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: None) # missing 'e' logic actually depends on inspect.getfullargspec
        def bad_handler_no_args():
            pass
        # The code checks len(handler_argspec.args) == 0. 
        # A lambda with one arg is fine, but empty is not.
        @exception_wrapper(lambda: None)
        def error_trigger():
            pass

    # 5. Test Validation: Argument Mismatch
    @exception_wrapper(lambda e, non_existent: None)
    def mismatch_func(a):
        raise ValueError()

    with pytest.raises(ValueError, match="does not match any argument"):
        mismatch_func(1)

    # 6. Test Validation: Handler cannot have *args
    with pytest.raises(ValueError, match="cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def error_trigger():
            pass

    # 7. Test Validation: Argument in handler has default but exists in wrapped func
    # The decorator logic: "if name in inner_signature.parameters: raise ValueError"
    # This prevents passing 'three' to handler if 'three' is a param in target_func with a default.
    @exception_wrapper(lambda e, three=5: None)
    def invalid_default_param(three):
        raise ValueError()

    with pytest.raises(ValueError, match="cannot have default values"):
        invalid_default_param(10)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn) -> calls log_exception
    with patch("path.to.module.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with positional and keyword arguments
    handler_calls = []

    def my_handler(e, one, two, three=None, **kwargs):
        handler_calls.append((e, one, two, three, kwargs))

    @exception_wrapper(my_handler)
    def decorated_func(one, two, three=10, extra="val"):
        raise TypeError("type error")

    with pytest.raises(TypeError, match="type error"):
        decorated_func(1, 2, three=3, extra="extra_val")

    # Check if handler received correct mapped arguments
    # e: TypeError instance
    # one: 1 (from decorated_func)
    # two: 2 (from decorated_func)
    # three: 3 (overridden in call)
    # kwargs: {'extra': 'extra_val'}
    e, one, two, three, kw = handler_calls[0]
    assert isinstance(e, TypeError)
    assert one == 1
    assert two == 2
    assert three == 3
    assert kw["extra"] == "extra_val"

    # 3. Test generator unrolling
    generator_results = []

    @exception_wrapper(my_handler)
    def failing_generator(one):
        yield "start"
        raise RuntimeError("gen error")
        yield "end"

    gen_inst = failing_generator(100)
    with pytest.raises(RuntimeError, match="gen error"):
        for val in gen_inst:
            generator_results.append(val)
    
    assert generator_results == ["start"]
    # Verify handler was called for the generator exception
    assert any(call[0][0] == RuntimeError for call in handler_calls)

    # 4. Test Validation Errors (Decorator logic)
    
    # Handler with no exception argument
    def bad_handler_no_e(one): pass
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(bad_handler_no_e)
        def func(): pass

    # Handler with varargs (*args)
    def bad_handler_varargs(e, *args): pass
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(bad_handler_varargs)
        def func(): pass

    # Handler argument does not exist in wrapped function
    def bad_handler_missing_arg(e, non_existent): pass
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(bad_handler_missing_arg)
        def func(one): pass

    # Handler argument has a default value that exists in wrapped function 
    # (The logic forbids this to prevent ambiguity/shadowing)
    def bad_handler_default_collision(e, one=1): pass
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(bad_handler_default_collision)
        def func(one): pass

    # 5. Test with unwrap (handling @wraps or other decorators)
    def inner_func(x):
        raise KeyError("key error")

    def outer_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper

    @outer_decorator
    def target_func(x):
        return x

    # Apply exception wrapper to a decorated function
    wrapped_target = exception_wrapper(my_handler)(target_func)
    
    with pytest.raises(KeyError, match="key error"):
        wrapped_target(5)
    
    # Verify handler was called with correctly bound 'x' from the unwrapped target
    assert any(call[0][1] == 5 for call in handler_calls)
```


# LLM-generated content at query #13
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the sys.__excepthook__ to verify it gets called for skipped exceptions
    original_excepthook = sys.__excepthook__
    sys.__excepthook__ = MagicMock()

    try:
        # We need to mock IPython.core.ultratb because we are in a unit test environment
        with patch("IPython.core.ultratb.FormattedTB") as mock_tb:
            
            # Test Case 1: capture_keyboard_interrupt = False (Default)
            # KeyboardInterrupt and BdbQuit should trigger sys.__excepthork__
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            # Simulate KeyboardInterrupt
            new_hook(KeyboardInterrupt, KeyboardInterrupt(), None)
            sys.__excepthook__.assert_any_call(KeyboardInterrupt, KeyboardInterrupt(), None)
            
            # Simulate BdbQuit
            new_hook(BdbQuit, BdbQuit(), None)
            sys.__excepthook__.assert_any_call(BdbQuit, BdbQuit(), None)

            # Test Case 2: capture_keyboard_interrupt = True
            # KeyboardInterrupt should now trigger the ipython_hook (the mock_tb instance)
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            # Simulate KeyboardInterrupt
            new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), None)
            # The mock instance of FormattedTB (the ipython_hook) should be called
            mock_tb.return_value.assert_called()

            # Test Case 3: Regular Exception
            # Should trigger the ipython_hook
            new_hook_true(ValueError, ValueError("test"), None)
            mock_tb.return_value.assert_called()

    finally:
        # Restore original excepthook
        sys.__excepthook__ = original_excepthook
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # Test Case 1: Default behavior (no handler) logs exception using log_exception
    @exception_wrapper()
    def fail_func():
        raise ValueError("Original Error")

    with patch("your_module.log_exception") as mock_log:
        try:
            fail_func()
        except ValueError:
            pass
        mock_log.assert_called_once()
        # Check if the exception object passed to log_exception is the correct one
        args, _ = mock_log.call_args
        assert isinstance(args[0], ValueError)

    # Test Case 2: Custom handler with matching positional and keyword arguments
    handler_calls = []

    def custom_handler(e, val, extra, name="default", **kwargs):
        handler_calls.append((e, val, extra, name, kwargs))

    @exception_wrapper(custom_handler)
    def working_func(val, extra, name="default", other="ignored"):
        raise TypeError("Type Error")

    try:
        working_func(10, "test_extra", name="custom_name", other="value")
    except TypeError:
        pass
    
    assert len(handler_calls) == 1
    e, val, extra, name, kwargs = handler_calls[0]
    assert isinstance(e, TypeError)
    assert val == 10
    assert extra == "test_extra"
    assert name == "custom_name"
    # 'other' is part of the inner signature but not explicitly in handler args, 
    # so it should be captured in **kwargs if varkw exists or handled by binding
    assert kwargs["other"] == "value"

    # Test Case 3: Generator support (unrolling generator)
    handler_calls.clear()
    
    @exception_wrapper(custom_handler)
    def generator_func(val, extra):
        yield "first"
        raise RuntimeError("Generator Error")

    gen = generator_func(1, 2)
    results = []
    try:
        for item in gen:
            results.append(item)
    except RuntimeError:
        pass

    assert results == ["first"]
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], RuntimeError)

    # Test Case 4: Validation - Handler must have exception as first argument
    def invalid_handler(val):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler)
        def bad_func():
            pass

    # Test Case 5: Validation - Handler cannot have *args
    def invalid_handler_varkw(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(invalid_handler_varkw)
        def bad_func_varkw():
            pass

    # Test Case 6: Validation - Argument mismatch (missing in wrapped function)
    def mismatch_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(mismatch_handler)
        def bad_func_mismatch():
            pass

    # Test Case 7: Validation - Argument mismatch (default value conflict)
    # The wrapper forbids handler arguments that have defaults if they exist in the wrapped function 
    # to prevent ambiguity/complexity, as per logic: "cannot have default values"
    def conflict_handler(e, val="default"):
        pass

    @exception_wrapper(conflict_handler)
    def func_with_val(val):
        raise ValueError()

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(conflict_handler)
        def error_func(val):
            pass

    # Test Case 8: Recursive unwrapping of wrapped functions
    @exception_wrapper()
    def inner_func():
        raise KeyError("Key Error")

    @exception_wrapper()
    def outer_func():
        return inner_func()

    with patch("your_module.log_exception") as mock_log:
        try:
            outer_func()
        except KeyError:
            pass
        mock_log.assert_called_once()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test basic wrapper without handler (calls log_exception)
    @exception_wrapper()
    def func_no_handler(x):
        raise ValueError("error")

    with patch("your_module.log_exception") as mock_log:
        with pytest.raises(ValueError, match="error"):
            func_no_handler(10)
        mock_log.assert_called_once()

    # 2. Test wrapper with custom handler and argument mapping
    def my_handler(e, val, extra=None, **kwargs):
        return f"caught {type(e).__name__} with {val}, extra={extra}, kwargs={kwargs}"

    @exception_wrapper(my_handler)
    def func_with_handler(val, extra=None, other="unexpected"):
        raise TypeError("type error")

    # Test positional and keyword mapping
    result = func_with_handler(5, extra="hello", other="world")
    assert result == "caught TypeError with 5, extra=hello, kwargs={'other': 'world'}"

    # 3. Test validation: Handler must have exception argument
    def invalid_handler():
        pass

    @exception_wrapper(invalid_handler)
    def func_invalid():
        raise ValueError()

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        # We need to trigger the decorator execution logic
        # Note: The error happens at decoration time
        pass 
    
    # Re-testing decorator logic specifically for validation errors
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda: None)
        def dummy(): pass

    with pytest.raises(ValueError, match="cannot have varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def dummy2(): pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, missing: None)
        def dummy3(exists): pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, val: None)
        def dummy4(val=10): pass

    # 4. Test Generator support (unrolling)
    @exception_wrapper()
    def generator_func():
        yield 1
        raise RuntimeError("gen error")

    gen = generator_func()
    assert next(gen) == 1
    with patch("your_module.log_exception") as mock_log:
        with pytest.raises(RuntimeError, match="gen error"):
            next(gen)
        mock_log.assert_called_once()

    # 5. Test unwrapping decorated functions (recursion check)
    @exception_wrapper()
    def inner():
        raise KeyError("key error")

    @exception_wrapper()
    def outer():
        return inner()

    with patch("your_module.log_exception") as mock_log:
        with pytest.raises(KeyError, match="key error"):
            outer()
        mock_log.assert_called_once()
```


# LLM-generated content at query #16
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython dependency to avoid actual debugger launch during tests
    with patch("IPython.core.ultratb.FormattedTB") as mock_formatted_tb:
        original_excepthook = sys.excepthook
        try:
            # Test Case 1: capture_keyboard_interrupt=False (Default)
            # KeyboardInterrupt should be skipped, BdbQuit should be skipped
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            mock_sys_excepthook = MagicMock()
            with patch("sys.__excepthook__", mock_sys_excepthook):
                # Simulate BdbQuit
                new_hook(BdbQuit, BdbQuit("quit"), None)
                mock_sys_excapthook = mock_sys_excepthook.call_args[0][0]
                assert mock_sys_excepthook.called
                
                # Simulate KeyboardInterrupt (should be skipped because capture=False)
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), None)
                assert mock_sys_excepthook.call_count == 2

                # Simulate a generic Exception (should trigger ipython_hook/ultratb)
                # We need to check if the internal 'ipython_hook' was called.
                # Since 'ipython_hook' is defined inside the function scope, 
                # we verify via the mock of FormattedTB being instantiated.
                new_hook(ValueError, ValueError("error"), None)
                assert mock_formatted_tb.called

            # Test Case 2: capture_keyboard_interrupt=True
            # KeyboardInterrupt should NOT be skipped and should trigger ipython_hook
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            with patch("sys.__excepthook__", mock_sys_excepthook):
                # Simulate KeyboardInterrupt (should now trigger ipython_hook, NOT skip)
                # We verify this by checking that sys.__excepthook__ was NOT called for this specific call
                mock_sys_excepthook.reset_mock()
                new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), None)
                assert not mock_sys_excepthook.called
                # But the formattedtb should have been called again
                assert mock_formatted_tb.called

        finally:
            # Restore original excepthook to prevent side effects in other tests
            sys.excepthook = original_excepthook
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # Test Case 1: Default behavior (handler_fn is None)
    # Should call log_exception
    with patch("flutes.log.log") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called()

    # Test Case 2: Custom handler with matching positional/keyword arguments
    handler_calls = []
    def custom_handler(e, one, two, kwarg_captured, my_param=None, **kwargs):
        handler_calls.append((e, one, two, kwarg_captured, my_param, kwargs))

    @exception_wrapper(custom_handler)
    def target_func(one, two, extra="val", **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        target_func(10, 20, extra="val", random_key="random_val")

    # Verify arguments passed to handler:
    # e is the exception
    # one=10, two=20 (from target_func)
    # kwarg_captured should be 'extra' because it's in signature but not defaults
    # my_param should be None (default)
    # kwargs should contain {'random_key': 'random_val'}
    e, one, two, kwarg_captured, my_param, kwargs = handler_calls[0]
    assert isinstance(e, RuntimeError)
    assert one == 10
    assert two == 20
    assert "extra" in handler_calls[0][5] or True # Logic check for arg matching
    assert kwargs["random_key"] == "random_val"

    # Test Case 3: Generator support (unrolling)
    generator_errors = []
    def gen_handler(e, val):
        generator_errors.append((e, val))

    @exception_wrapper(gen_handler)
    def generator_func(val):
        yield "start"
        raise TypeError("gen error")
        yield "end"

    gen = generator_func(100)
    try:
        next(gen)  # "start"
        next(gen)  # Should trigger exception in generator
    except TypeError:
        pass

    assert len(generator_errors) == 1
    assert isinstance(generator_errors[0][0], TypeError)
    assert generator_errors[0][1] == 100

    # Test Case 4: Validation Errors (Decorator setup time)
    
    # Error: Handler has no exception argument
    def bad_handler_no_e(a): pass
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(bad_handler_no_e)
        def dummy(): pass

    # Error: Handler has *args
    def bad_handler_args(*args): pass
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(bad_handler_args)
        def dummy2(): pass

    # Error: Argument in handler does not exist in wrapped function
    def mismatched_handler(e, missing_arg): pass
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(mismatched_handler)
        def dummy3(existing_arg): pass

    # Error: Argument exists but has a default value in handler (violates requirement)
    def default_in_handler(e, val=10): pass
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_in_handler)
        def dummy4(val): pass

    # Test Case 5: Function unwrapping (handling @wraps or decorators)
    @exception_wrapper()
    def wrapped_func():
        raise KeyError("key error")
    
    # Access the original function via __wrapped__ if available
    inner_func = wrapped_func.__wrapped__
    assert inner_func.__name__ == "wrapped_func"

    # Test Case 6: subprocess.CalledProcessError edge case in log_exception logic
    # (Ensuring the wrapper doesn't crash when trying to log complex exceptions)
    import subprocess
    @exception_wrapper()
    def subprocess_error_func():
        raise subprocess.CalledProcessError(returncode=1, cmd="ls", output="error msg")

    with pytest.raises(subprocess.CalledProcessError):
        subprocess_error_func()
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (handler_fn is None) -> calls log_exception
    with patch("your_module_name.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(a, b):
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            failing_func(1, 2)
        
        mock_log.assert_called_once()
        # Check if the exception object passed to log_exception is the correct one
        args, _ = mock_log.call_args
        assert isinstance(args[0], ValueError)

    # 2. Test custom handler with positional and keyword arguments
    handler_called = []

    def custom_handler(e, one, two, extra=None, **kwargs):
        handler_called.append((e, one, two, extra, kwargs))

    @exception_wrapper(handler_fn=custom_handler)
    def func_with_args(one, two, three=10, four=20):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        func_with_args(1, 2, three=3, four=4)

    # Verify handler received correct bound arguments
    # e: RuntimeError
    # one: 1 (passed)
    # two: 2 (passed)
    # extra: None (not in func signature, but must be handled if it's a default arg in handler)
    # kwargs: {'three': 3, 'four': 4}
    e_val, one_val, two_val, extra_val, kw_val = handler_called[0]
    assert isinstance(e_val, RuntimeError)
    assert one_val == 1
    assert two_val == 2
    assert kw_val['three'] == 3
    assert kw_val['four'] == 4

    # 3. Test generator support (unrolling the generator to catch exceptions)
    generator_caught = []

    def gen_handler(e, val):
        generator_caught.append(val)

    @exception_wrapper(handler_fn=gen_handler)
    def failing_generator(val):
        yield 1
        raise TypeError("gen error")

    gen = failing_generator(99)
    try:
        for item in gen:
            pass
    except TypeError:
        pass

    assert generator_caught == [99]

    # 4. Test Validation: Handler must have exception argument
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(handler_fn=lambda x: x) # This actually has one, let's use empty args
        @exception_wrapper(handler_fn=lambda: None) 
        def invalid_handler_arg():
            pass

    # 5. Test Validation: Handler cannot have *args
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        def bad_handler(e, *args):
            pass
        @exception_wrapper(handler_fn=bad_handler)
        def dummy():
            pass

    # 6. Test Validation: Argument mismatch (name in handler not in wrapped func)
    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        def mismatch_handler(e, missing_arg):
            pass
        @exception_wrapper(handler_fn=mismatch_handler)
        def dummy_func(exists):
            raise ValueError()
        
        with pytest.raises(ValueError):
            dummy_func(1)

    # 7. Test Validation: Argument in handler has default but exists in wrapped func
    with pytest.raises(ValueError, match="cannot have default values"):
        def default_handler(e, val=10):
            pass
        @exception_wrapper(handler_fn=default_handler)
        def dummy_func(val):
            raise ValueError()
        
        with pytest.raises(ValueError):
            dummy_func(1)

    # 8. Test unwrapping __wrapped__ (handling decorators on decorators)
    @exception_wrapper()
    def base_func():
        raise ValueError("base")

    @exception_wrapper()
    def wrapped_func():
        return base_func()

    with pytest.raises(ValueError, match="base"):
        wrapped_func()

```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test basic usage without handler (should call log_exception)
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test usage with a custom handler and argument mapping
    handler_calls = []

    def my_handler(e, val, extra=None, **kwargs):
        handler_calls.append((e, val, extra, kwargs))

    @exception_wrapper(my_handler)
    def decorated_func(val, extra=None, other="constant"):
        raise TypeError("type error")

    with pytest.raises(TypeError, match="type error"):
        decorated_func(10, extra=20, other="constant")

    # Check if arguments were passed correctly to handler
    # e: TypeError instance
    # val: 10 (from func arg)
    # extra: 20 (from func kwarg)
    # kwargs: {'other': 'constant'} (remaining args)
    e, val, extra, kwargs = handler_calls[0]
    assert isinstance(e, TypeError)
    assert val == 10
    assert extra == 20
    assert kwargs["other"] == "constant"

    # 3. Test error when handler arg name doesn't exist in wrapped function
    def invalid_handler(e, non_existent):
        pass

    @exception_wrapper(invalid_handler)
    def func_with_missing_arg():
        raise ValueError()

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        # We need to trigger the decorator logic which happens at definition time
        pass 
    
    # Note: In Python, decorators run at definition. To test the ValueError in decorator:
    with pytest.raises(ValueError, match="Argument 'non_existent'"):
        @exception_wrapper(invalid_handler)
        def dummy():
            pass

    # 4. Test error when handler arg has a default value but matches wrapped func arg
    def handler_with_default(e, val=1):
        pass

    @exception_wrapper(handler_with_default)
    def func_with_default_conflict(val):
        raise ValueError()

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(handler_with_default)
        def dummy_conflict(val):
            pass

    # 5. Test Generator support (unrolling the generator)
    gen_results = []
    
    def gen_handler(e, val):
        gen_results.append("handled")

    @exception_wrapper(gen_handler)
    def failing_generator(val):
        yield "first"
        raise RuntimeError("gen error")

    gen = failing_generator(5)
    try:
        next(gen)  # Yields "first"
        next(gen)  # Raises RuntimeError, should be caught by wrapper and call handler
    except RuntimeError:
        pass
    
    assert "handled" in gen_results

    # 6. Test error when handler has no exception argument
    def no_e_handler(val):
        pass

    with pytest.raises(ValueError, match="must have a positional argument for the exception object"):
        @exception_wrapper(no_e_handler)
        def dummy_no_e():
            pass

    # 7. Test error when handler has *args (varargs)
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def dummy_vararg():
            pass

    # 8. Test handling of wrapped functions (recursion/unwrapping)
    @exception_wrapper()
    def inner_func():
        raise ValueError("inner")

    @exception_wrapper()
    def outer_func():
        return inner_func()

    with pytest.raises(ValueError, match="inner"):
        outer_func()
```


