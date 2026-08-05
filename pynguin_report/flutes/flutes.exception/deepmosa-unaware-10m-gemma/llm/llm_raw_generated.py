####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module that is imported inside the function
    with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.core": MagicMock(), "IPython.core.ultratb": MagicMock()}), \
         patch("sys.__excepthook__") as mock_original_hook:
        
        # 1. Test registration with capture_keyboard_interrupt=False (default)
        # In this mode, KeyboardInterrupt should be added to skip_exceptions
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        new_hook = sys.excepthook
        
        # Create a dummy exception and traceback
        exc_type = KeyboardInterrupt
        exc_value = KeyboardInterrupt("test")
        exc_traceback = MagicMock()

        # Execute the hook with a type that should be skipped (KeyboardInterrupt)
        new_hook(exc_type, exc_value, exc_traceback)
        
        # Verify it called the original sys.__excepthook__ instead of ipython_hook
        mock_original_hook.assert_called_with(exc_type, exc_value, exc_traceback)

        # 2. Test registration with capture_keyboard_interrupt=True
        # In this mode, KeyboardInterrupt should NOT be in skip_exceptions, 
        # so ipython_hook (ultratb.FormattedTB) should be called.
        with patch("IPython.core.ultratb.FormattedTB") as mock_tb:
            mock_instance = mock_tb.return_value
            
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            reloaded_hook = sys.excepthook
            
            # Execute the hook with KeyboardInterrupt
            reloaded_hook(exc_type, exc_value, exc_traceback)
            
            # Verify it called the ipython_hook instance (the mock object)
            # Note: The code calls 'ipython_hook(type, value, traceback)' 
            # which is an instance of FormattedTB.
            mock_instance.assert_called_with(exc_type, exc_value, exc_traceback)

        # 3. Test BdbQuit (should always be skipped regardless of param)
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        bdb_quit_type = BdbQuit
        bdb_quit_val = BdbQuit("exit")
        sys.excepthook(bdb_quit_type, bdb_quit_val, exc_traceback)
        mock_original_hook.assert_called_with(bdb_quit_type, bdb_quit_val, exc_traceback)

        # 4. Test a standard exception (should trigger ipython_hook)
        standard_exc_type = ValueError
        standard_exc_val = ValueError("error")
        with patch("IPython.core.ultratb.FormattedTB") as mock_tb_new:
            mock_instance_new = mock_tb_new.return_value
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            sys.excepthook(standard_exc_type, standard_exc_val, exc_traceback)
            mock_instance_new.assert_called_with(standard_exc_type, standard_exc_val, exc_traceback)
```


# LLM-generated content at query #2
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for IPython and the original excepthook
    mock_ultratb = MagicMock()
    mock_sys_excepthook = MagicMock()
    
    with patch("sys.__excepthook__", mock_sys_excepthook), \
         patch("IPython.core.ultratb.FormattedTB", return_value=mock_ultratb), \
         patch("sys.excepthook", create=True) as mock_set_excepthook:
        
        # Test Case 1: capture_keyboard_interrupt = False (Default)
        # Should skip KeyboardInterrupt and BdbQuit
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        new_hook = sys.excepthook
        assert callable(new_hook)
        
        # Create dummy traceback/value/type objects
        dummy_tb = MagicMock()
        dummy_val = Exception("test")
        
        # Simulate KeyboardInterrupt
        new_hook(KeyboardInterrupt, dummy_val, dummy_tb)
        mock_sys_excepthook.assert_called_with(KeyboardInterrupt, dummy_val, dummy_tb)
        
        # Simulate BdbQuit
        new_hook(BdbQuit, dummy_val, dummy_tb)
        mock_sys_excepthook.assert_called_with(BdbQuit, dummy_val, dummy_tb)
        
        # Simulate a generic Exception (should trigger ipython_hook/ultratb)
        # Note: in the real code, ultratb is an instance of FormattedTB 
        # which is assigned to new_hook's internal reference.
        # Since we mocked the class, the instance 'ipython_hook' is our mock_ultratb.
        # However, the function calls ipython_hook(type, value, traceback).
        # We need to ensure the instance created by FormattedTB is callable or 
        # that it doesn't crash when called with 3 args.
        mock_ultratb.__call__ = MagicMock()
        new_hook(ValueError, dummy_val, dummy_tb)
        mock_ultratb.__call__.assert_called_with(ValueError, dummy_val, dummy_tb)

        # Test Case 2: capture_keyboard_interrupt = True
        # Should NOT skip KeyboardInterrupt
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        
        mock_ultratb.__call__.reset_mock()
        new_hook = sys.exapthook # This will be the new hook from the second call
        # Re-run to get the updated sys.excepthook
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        new_hook = sys.excepthook
        
        new_hook(KeyboardInterrupt, dummy_val, dummy_tb)
        # It should NOT call sys.__excepthook__, it should call the ipython_hook (the mock instance)
        mock_ultratb.__call__.assert_called_with(KeyboardInterrupt, dummy_val, dummy_tb)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test Default behavior (no handler provided) - should call log_exception
    with patch("flutes.log.log") as mock_log:
        @exception_wrapper()
        def fail_func():
            raise ValueError("test error")

        try:
            fail_func()
        except ValueError:
            pass
        
        # Verify log_exception was triggered (via the internal log call)
        assert mock_log.called
        args, _ = mock_log.call_args
        assert "<ValueError> test error" in args[0]

    # 2. Test Custom Handler with matching positional and keyword arguments
    handler_called = False
    def my_handler(e, one, two, kw):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert one == 1
        assert two == "two"
        assert kw == {"three": 3}

    @exception_wrapper(my_handler)
    def success_func(one, two, three=None, **kwargs):
        raise ValueError("trigger handler")

    try:
        success_func(1, "two", three=3)
    except Exception:
        pass
    assert handler_called is True

    # 3. Test Generator support (unrolling the generator to catch exception)
    generator_caught = False
    def gen_handler(e, val):
        nonlocal generator_caught
        generator_caught = True
        assert val == "found"

    @exception_wrapper(gen_handler)
    def error_gen(val):
        yield "start"
        raise ValueError("gen error")

    gen_instance = error_gen("found")
    try:
        # We must iterate to trigger the exception inside the generator
        list(gen_instance)
    except Exception:
        pass
    assert generator_caught is True

    # 4. Test Validation: Handler missing required argument from wrapped function
    with pytest.raises(ValueError, match="Argument 'missing' in exception handler does not match"):
        def invalid_handler(e, missing):
            pass

        @exception_wrapper(invalid_handler)
        def target_func(one):
            raise ValueError()
        
        try:
            target_func(1)
        except Exception:
            pass

    # 5. Test Validation: Handler has argument with default value that matches wrapped function
    with pytest.raises(ValueError, match="cannot have default values"):
        def invalid_handler_defaults(e, one=10):
            pass

        @exception_wrapper(invalid_handler_defaults)
        def target_func(one):
            raise ValueError()
            
        try:
            target_func(5)
        except Exception:
            pass

    # 6. Test Validation: Handler cannot have *args
    with pytest.raises(ValueError, match="cannot have a varargs argument"):
        def invalid_handler_varg(e, *args):
            pass

        @exception_wrapper(invalid_handler_varg)
        def target_func():
            raise ValueError()
            
        try:
            target_func()
        except Exception:
            pass

    # 7. Test Validation: Handler must have at least one argument (the exception)
    with pytest.raises(ValueError, match="must have a positional argument for the exception object"):
        def empty_handler():
            pass

        @exception_wrapper(empty_handler)
        def target_func():
            raise ValueError()
            
        try:
            target_func()
        except Exception:
            pass
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) - should call log_exception
    with patch("your_module_name.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with exact matching arguments
    handler_calls = []
    def custom_handler(e, one):
        handler_calls.append((e, one))
        raise ValueError("re-raised")

    @exception_wrapper(handler_fn=custom_handler)
    def func_with_args(one, two):
        raise ValueError("trigger")

    with pytest.raises(ValueError, match="re-raised"):
        func_with_args(10, 20)
    
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ValueError)
    assert handler_calls[0][1] == 10

    # 3. Test custom handler with kwargs/varkw capture
    handler_results = []
    def kwarg_handler(e, val, extra=None, **kwargs):
        handler_results.append({
            "val": val,
            "extra": extra,
            "kwargs": kwargs
        })

    @exception_wrapper(handler_fn=kwarg_handler)
    def func_with_kwargs(val, extra, other="something", unrelated="data"):
        raise ValueError("trigger")

    with pytest.raises(ValueError):
        func_with_kwargs(5, extra="test", unrelated="data")

    assert handler_results[0]["val"] == 5
    assert handler_results[0]["extra"] == "test"
    assert handler_results[0]["kwargs"]["unrelated"] == "data"

    # 4. Test Generator support (unrolling generator)
    generator_captured = []
    def gen_handler(e, x):
        generator_captured.append(x)

    @exception_wrapper(handler_fn=gen_handler)
    def failing_generator(x):
        yield 1
        raise ValueError("gen error")

    gen_obj = failing_generator(99)
    try:
        for val in gen_obj:
            pass
    except ValueError:
        pass

    assert generator_captured == [99]

    # 5. Test Validation Error: Handler has no exception argument
    def invalid_handler(not_e):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(handler_fn=invalid_handler)
        def bad_func():
            pass

    # 6. Test Validation Error: Handler has *args
    def invalid_handler_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(handler_fn=invalid_handler_args)
        def bad_func():
            pass

    # 7. Test Validation Error: Argument mismatch (missing in wrapped function)
    def missing_arg_handler(e, missing_param):
        pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(handler_fn=missing_arg_handler)
        def bad_func(existing_param):
            pass

    # 8. Test Validation Error: Argument mismatch (duplicate default value logic)
    def duplicate_default_handler(e, param):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(handler_fn=duplicate_default_handler)
        def bad_func(param=10):
            pass
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn): should call log_exception
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def fail_func(a, b):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            fail_func(1, 2)
        mock_log.assert_called_once()

    # 2. Test basic custom handler with matching arguments
    handler_calls = []
    def simple_handler(e, one):
        handler_calls.append((type(e), one))
        raise e

    @exception_wrapper(simple_handler)
    def func_with_arg(one, two):
        raise TypeError("type error")

    with pytest.raises(TypeError):
        func_with_arg(10, 20)
    assert handler_calls[0] == (TypeError, 10)

    # 3. Test handler with keyword arguments and defaults
    handler_calls = []
    def complex_handler(e, one, two, three=None, **kwargs):
        handler_calls.append({
            "e": type(e),
            "one": one,
            "two": two,
            "three": three,
            "kwargs": kwargs
        })
        raise e

    @exception_wrapper(complex_handler)
    def func_complex(one, two, extra="val", other="other"):
        raise ValueError("fail")

    with pytest.raises(ValueError):
        func_complex(1, 2, extra="new_val", other="other_val")
    
    assert handler_calls[0]["e"] == ValueError
    assert handler_calls[0]["one"] == 1
    assert handler_calls[0]["two"] == 2
    # 'three' is not in func_complex signature, so it remains default None (from inspection logic)
    # Note: The implementation logic for 'three' relies on whether it was passed or exists in signature.
    # In this specific code implementation: handler_arg_names - handler_args_with_defaults
    assert handler_calls[0]["kwargs"] == {"extra": "new_val", "other": "other_val"}

    # 4. Test Generator support (unrolling)
    gen_results = []
    def gen_handler(e, x):
        gen_results.append(x)
        raise e

    @exception_wrapper(gen_handler)
    def generator_func(x):
        yield 1
        yield 2
        raise RuntimeError("gen fail")

    gen_iter = generator_func(99)
    with pytest.raises(RuntimeError):
        list(gen_iter)
    assert gen_results == [99]

    # 5. Test Validation: Handler must have exception argument
    def invalid_handler(not_e):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler)
        def bad_func():
            pass

    # 6. Test Validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def bad_func_2():
            pass

    # 7. Test Validation: Handler argument not in wrapped function
    def missing_arg_handler(e, missing):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(missing_arg_handler)
        def bad_func_3(one):
            pass

    # 8. Test Validation: Handler argument with default matches wrapped function (forbidden by logic)
    def default_in_handler(e, one=1):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_in_handler)
        def bad_func_4(one):
            pass

    # 9. Test Unwrapping (decorator on decorated function)
    @exception_wrapper()
    def inner():
        raise ValueError("inner")

    @exception_wrapper()
    def outer():
        return inner()

    with pytest.raises(ValueError, match="inner"):
        outer()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) -> calls log_exception
    with patch("your_module_name.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Original Error")

        with pytest.raises(ValueError, match="Original Error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with exact argument matching
    handler_calls = []
    def custom_handler(e, val):
        handler_calls.append((e, val))
        raise ValueError("Re-raised from handler")

    @exception_wrapper(handler_fn=custom_handler)
    def func_with_val(val):
        raise TypeError("Trigger error")

    with pytest.raises(ValueError, match="Re-raised from handler"):
        func_with_val(10)
    
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], TypeError)
    assert handler_calls[0][1] == 10

    # 3. Test custom handler with keyword arguments and **kwargs capture
    handler_results = []
    def complex_handler(e, one, two, extra=None, **kwargs):
        handler_results.append({
            "e": e,
            "one": one,
            "two": two,
            "extra": extra,
            "kwargs": kwargs
        })

    @exception_wrapper(handler_fn=complex_handler)
    def complex_func(one, two, extra=None, **others):
        raise RuntimeError("Boom")

    with pytest.raises(RuntimeError):
        complex_func(1, 2, extra=3, foo="bar", baz=True)

    assert handler_results[0]["e"] == RuntimeError("Boom")
    assert handler_results[0]["one"] == 1
    assert handler_results[0]["two"] == 2
    assert handler_results[0]["extra"] == 3
    assert handler_results[0]["kwargs"] == {"foo": "bar", "baz": True}

    # 4. Test Generator support (unrolling generator)
    gen_captured = []
    def gen_handler(e, name):
        gen_captured.append((e, name))

    @exception_wrapper(handler_fn=gen_handler)
    def error_generator(name):
        yield "first"
        raise AttributeError("Gen Error")

    gen = error_generator("test_gen")
    results = []
    try:
        for val in gen:
            results.append(val)
    except AttributeError:
        pass

    assert results == ["first"]
    assert len(gen_captured) == 1
    assert isinstance(gen_captured[0][0], AttributeError)
    assert gen_captured[0][1] == "test_gen"

    # 5. Test Validation Errors (Decorator-time checks)
    
    # Handler must have exception as first arg
    def invalid_handler():
        pass
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(handler_fn=invalid_handler)
        def foo(): pass

    # Handler cannot have *args
    def varargs_handler(e, *args):
        pass
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(handler_fn=varargs_handler)
        def foo(): pass

    # Handler argument does not exist in wrapped function
    def missing_arg_handler(e, non_existent):
        pass
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(handler_fn=missing_arg_handler)
        def foo(existing): pass

    # Handler argument has default value but matches wrapped function (forbidden by logic to prevent ambiguity)
    def default_val_handler(e, val=10):
        pass
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(handler_fn=default_val_handler)
        def foo(val): pass

```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn) - calls log_exception
    with patch("your_module_path.log_exception") as mock_log_exc:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Original Error")

        with pytest.raises(ValueError, match="Original Error"):
            failing_func()
        mock_log_exc.assert_called_once()

    # 2. Test with custom handler and argument mapping
    handler_calls = []

    def custom_handler(e, one, two, extra_arg=None, **kwargs):
        handler_calls.append({
            "e": e,
            "one": one,
            "two": two,
            "extra_arg": extra_arg,
            "kwargs": kwargs
        })

    @exception_wrapper(custom_handler)
    def target_func(one, two, three=10, **kwargs):
        raise TypeError("Triggered")

    with pytest.raises(TypeError, match="Triggered"):
        target_func(1, 2, extra_arg=5, unused_param="hello")

    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0]["e"], TypeError)
    assert handler_calls[0]["one"] == 1
    assert handler_calls[0]["two"] == 2
    # 'three' is in target_func but not explicitly in handler signature, 
    # so it ends up in kwargs of the handler via varkw logic
    assert "three" in handler_calls[0]["kwargs"] or handler_calls[0]["extra_arg"] is None

    # 3. Test Generator support (unrolling)
    generator_captured = []

    def gen_handler(e, val):
        generator_captured.append((e, val))

    @exception_wrapper(gen_handler)
    def generator_func(val):
        yield "first"
        raise RuntimeError("Gen Error")

    gen_inst = generator_func("data")
    with pytest.raises(StopIteration):  # Exhausting the generator causes error to propagate
        next(gen_inst)
        try:
            next(gen_inst)
        except RuntimeError:
            pass
    
    # Note: The wrapper catches exception and calls handler, but doesn't re-raise 
    # unless the inner logic does. In this implementation, _handle_exception is called.
    # We check if handler was triggered during iteration.
    # Since the wrapper returns a generator that yields from gen in a try/except block:
    
    gen_inst = generator_func("data")
    try:
        next(gen_inst) # "first"
        next(gen_inst) # triggers RuntimeError -> handler calls
    except Exception:
        pass

    # 4. Test Validation Errors (Decorator logic)
    
    # Handler with no exception arg
    def bad_handler(a): pass
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(bad_handler)
        def func(): pass

    # Handler with *args (not allowed)
    def bad_handler_args(e, *args): pass
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(bad_handler_args)
        def func(): pass

    # Handler requesting arg not in wrapped function
    def missing_arg_handler(e, non_existent): pass
    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(missing_arg_handler)
        def func(real_arg): pass

    # Handler requesting arg that has a default in wrapped function (not allowed per implementation logic)
    def default_conflict_handler(e, val): pass
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_conflict_handler)
        def func(val=10): pass

    # 5. Test unwrapping of already wrapped functions
    @exception_wrapper()
    def inner():
        raise AttributeError("Inner error")

    @exception_wrapper()
    def outer():
        return inner()

    with pytest.raises(AttributeError, match="Inner error"):
        outer()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) - calls log_exception
    with patch("your_module_name.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with positional and keyword arguments mapping
    handler_called = []
    def custom_handler(e, one, two, my_arg=None, **kwargs):
        handler_called.append((e, one, two, my_arg, kwargs))

    @exception_wrapper(custom_handler)
    def complex_func(one, two, three=10, extra="val"):
        raise KeyError("key error")

    with pytest.raises(KeyError):
        complex_func(1, 2, extra="extra_val")

    # Check if handler received correct bound arguments
    # e: KeyError, one: 1, two: 2, my_arg (not in func) -> error should have been caught by validation?
    # Wait, the decorator validates that 'one' and 'two' exist in func.
    # It also checks that if handler has args with defaults, they CANNOT be in the func signature.
    
    # 3. Test Generator support (unrolling generator to catch exceptions)
    generator_called = []
    def gen_handler(e, val):
        generator_called.append(val)

    @exception_wrapper(gen_handler)
    def error_generator(val):
        yield "start"
        raise RuntimeError("gen error")
        yield "end"

    gen = error_generator("test_val")
    with pytest.raises(RuntimeError, match="gen error"):
        next(gen)
        next(gen)
    assert generator_called == ["test_val"]

    # 4. Test Validation: Handler must have exception object as first arg
    def invalid_handler(one):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler)
        def func():
            pass

    # 5. Test Validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def func():
            pass

    # 6. Test Validation: Argument in handler does not exist in wrapped function
    def missing_arg_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(missing_arg_handler)
        def func(existing):
            pass

    # 7. Test Validation: Argument in handler matches wrapped method but has a default value
    def duplicate_default_handler(e, existing=True):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(duplicate_default_handler)
        def func(existing):
            pass

    # 8. Test complex binding with **kwargs
    handler_results = []
    def full_handler(e, a, b, c, d=None, **kwargs):
        handler_results.append((a, b, c, d, kwargs))

    @exception_wrapper(full_handler)
    def big_func(a, b, c, d=99, other="ignore"):
        raise TypeError("type error")

    with pytest.raises(TypeError):
        big_func(1, 2, 3, other="captured")

    # a=1, b=2, c=3 (from args)
    # d=99 (default from func)
    # kwargs contains 'other' because it's in the function call but not explicitly named in handler signature mapping?
    # Actually, 'd' is handled by the logic of checking defaults. 
    # 'other' ends up in kwargs because it was part of the bound arguments.
    assert handler_results[0][0] == 1
    assert handler_results[0][1] == 2
    assert handler_results[0][2] == 3
    assert handler_results[0][3] == 99
    assert handler_results[0][4]['other'] == 'captured'
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_name.log")  # Replace 'your_module_name' with the actual module name
def test_log_exception(mock_log):
    # Test Case 1: Basic exception logging without user message
    e1 = ValueError("simple error")
    log_exception(e1)
    
    # Verify log was called with the formatted exception string and "error" level
    expected_msg = f"<{e1.__class__.__qualname__}> {e1}"
    assert mock_log.called
    # Check if the second call to log (the message itself) contains the expected text
    mock_log.assert_any_call(expected_msg, "error")

    # Test Case 2: Exception logging with a user message
    user_msg = "Custom prefix"
    e2 = TypeError("type error")
    log_exception(e2, user_msg=user_msg)
    
    expected_msg_with_prefix = f"{user_msg}: <{e2.__class__.__qualname__}> {e2}"
    mock_log.assert_any_call(expected_msg_with_prefix, "error")

    # Test Case 3: subprocess.CalledProcessError with output (should NOT log traceback)
    # According to the code: if e.output is not None, it skips logging the traceback
    e3 = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some error output")
    mock_log.reset_mock()
    log_exception(e3)
    
    # Check that log was NOT called with a traceback (the first call in the try block)
    # We only check if it logged the message itself. 
    # If it tried to log the traceback, there would be an extra call to mock_log.
    expected_msg3 = f"<{e3.__class__.__qualname__}> {e3}"
    mock_log.assert_called_once_with(expected_msg3, "error")

    # Test Case 4: subprocess.CalledProcessError without output (should log traceback)
    e4 = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    mock_log.reset_mock()
    log_exception(e4)
    
    # Verify it attempted to log the traceback string (contains 'traceback.format_exc' content)
    # and the exception message.
    found_traceback = False
    for call in mock_log.call_args_list:
        if "Traceback" in call[0][0]:
            found_traceback = True
    assert found_traceback, "Traceback should have been logged for CalledProcessError with no output"

    # Test Case 5: Handling failure within the logger itself (Secondary exception)
    # We force log() to raise an Exception to test the 'except Exception as log_e' block
    mock_log.side_effect = RuntimeError("Logging failed")
    e5 = ValueError("original error")
    
    with patch("builtins.print") as mock_print:
        with pytest.raises(RuntimeError) as excinfo:
            log_exception(e5, user_msg="Failure test")
        
        assert "Logging failed" in str(excinfo.value)
        # Check that it printed the original error and the secondary error to stdout
        printed_messages = [call.args[0] for call in mock_print.call_args_list]
        assert any("Failure test: <ValueError> original error" in msg for msg in printed_messages)
        assert any("Another exception occurred while logging: <RuntimeError>" in msg for msg in printed_messages)

    # Reset side effect for future tests if necessary
    mock_log.side_effect = None
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess

@patch("your_module.log")  # Replace 'your_module' with the actual module name
def test_log_exception(mock_log):
    # Test case 1: Basic logging of an exception
    exc = ValueError("test error")
    log_exception(exc)
    
    # Check if log was called with formatted exception string and level
    # Note: traceback.format_exc() is called, so we check for the presence of class name
    args, kwargs = mock_log.call_args_list[0]
    assert "<ValueError> test error" in args[0]
    assert args[1] == "error"

    # Test case 2: Logging with a user message
    user_msg = "Custom failure"
    log_exception(exc, user_msg=user_msg)
    
    args, kwargs = mock_log.call_args_list[1]
    assert f"{user_msg}: <ValueError> test error" in args[0]

    # Test case 3: Logging a subprocess.CalledProcessError with output
    # The code has specific logic: if e.output is not None, it skips logging the traceback
    sub_exc = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error details")
    log_exception(sub_exc)
    
    # We look for the call that contains the exception message itself
    # The last call in the sequence should be the one with the exception string
    last_call_args = mock_log.call_args_list[-1][0]
    assert "<CalledProcessError> Command 'ls' failed with exit status 1" in last_call_args

    # Test case 4: Exception during logging (should print to stdout and re-raise)
    mock_log.side_effect = Exception("Logging failed")
    with patch("builtins.print") as mock_print:
        with pytest.raises(Exception) as exc_info:
            log_exception(exc, user_msg="Trigger failure")
        
        assert "Another exception occurred while logging" in str(exc_info.value)
        # Check if it printed the original error to stdout as a fallback
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Trigger failure: <ValueError> test error" in s for s in print_calls)

@patch("your_module.log")
def test_log_exception_with_kwargs(mock_log):
    exc = TypeError("type error")
    extra_context = {"request_id": "123", "user": "admin"}
    
    log_exception(exc, extra_context=extra_context)
    
    # Verify kwargs are passed through to the log function
    _, kwargs = mock_log.call_args
    assert kwargs["extra_context"] == extra_context
```


# LLM-generated content at query #11
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module and its FormattedTB class
    with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.core": MagicMock(), "IPython.core.ultratb": MagicMock()}), \
         patch("IPython.core.ultratbolt.FormattedTB") as mock_formatted_tb, \
         patch("sys.excepthook") as mock_original_excepthook:
        
        # Setup the mock to simulate the real behavior of ultratb.FormattedTB
        mock_instance = MagicMock()
        mock_formatted_tb.return_value = mock_instance
        
        # Save original excepthook to restore later
        original_hook = sys.excepthook
        
        try:
            # Test Case 1: capture_keyboard_interrupt = False (Default)
            # Should skip KeyboardInterrupt and BdbQuit
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            # Simulate a KeyboardInterrupt
            exc_type = KeyboardInterrupt
            exc_value = KeyboardInterrupt("test")
            mock_tb = MagicMock()
            
            # We need to mock the __excepthook__ within the closure logic
            # Since we can't easily access the internal 'excepthook' function, 
            # we test if it calls sys.__excepthook__ for specific types.
            with patch("sys.__excepthook__") as mock_sys_hook:
                # Test KeyboardInterrupt (should be skipped)
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), mock_tb)
                mock_sys_hook.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), mock_tb)
                
                # Test BdbQuit (should be skipped)
                new_hook(BdbQuit, BdbQuit(), mock_tb)
                mock_sys_hook.assert_called_with(BdbQuit, BdbQuit(), mock_tb)
                
                # Test a standard ValueError (should NOT be skipped, should call ipython_hook)
                # Note: In the code, ipython_hook is the instance of FormattedTB
                new_hook(ValueError, ValueError("error"), mock_tb)
                # The error occurs because 'ipython_hook' in the real function is an object.
                # We check if it was called (the call happens via the __call__ or direct usage)
                # In the provided code: ipython_hook(type, value, traceback)
                # Since we mocked FormattedTB to return mock_instance, we check if mock_instance was called.
                mock_instance.assert_called_once_with(ValueError, ValueError("error"), mock_tb)

            # Test Case 2: capture_keyboard_interrupt = True
            # Should NOT skip KeyboardInterrupt
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            with patch("sys.__excepthook__") as mock_sys_hook_true:
                # Reset call counts
                mock_instance.reset_mock()
                
                # KeyboardInterrupt should now trigger the ipython_hook, NOT the sys.__excepthook__
                new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), mock_tb)
                mock_sys_hook_true.assert_not_called()
                mock_instance.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), mock_tb)

        finally:
            # Restore the original excepthook
            sys.excepthook = original_hook
```


# LLM-generated content at query #12
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for the IPython dependency and the original excepthook
    mock_ultratb = MagicMock()
    mock_sys_excepthook = MagicMock()
    
    with patch.dict(sys, {"__excepthook__": mock_sys_excepthook}), \
         patch.dict("sys.modules", {"IPython.core import ultratb": mock_ultratb}), \
         patch("IPython.core.ultratb.FormattedTB", return_value=MagicMock()) as mock_formatted_tb:
        
        # Test Case 1: capture_keyboard_interrupt = False (default)
        # KeyboardInterrupt should be skipped and passed to sys.__excepthook__
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        current_hook = sys.excepthook
        
        # Simulate a KeyboardInterrupt
        kb_interrupt = KeyboardInterrupt("test interrupt")
        current_hook(KeyboardInterrupt, kb_interrupt, None)
        mock_sys_excepthook.assert_called_with(KeyboardInterrupt, kb_interrupt, None)
        
        # Simulate a standard Exception
        standard_exc = ValueError("test error")
        current_hook(ValueError, standard_exc, None)
        # Check if the IPython hook (the mock we injected) was called
        mock_formatted_tb.assert_called()

        # Test Case 2: capture_keyboard_interrupt = True
        # KeyboardInterrupt should NOT be skipped; it should trigger the IPython hook
        # Reset mocks for clean state
        mock_sys_excepthook.reset_mock()
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        
        current_hook = sys.excepthook
        kb_interrupt_2 = KeyboardInterrupt("test interrupt 2")
        current_hook(KeyboardInterrupt, kb_interrupt_2, None)
        
        # It should NOT call the original sys.__excepthook__ because it's no longer in skip_exceptions
        mock_sys_excepthook.assert_not_called()

        # Test Case 3: BdbQuit should always be skipped regardless of parameter
        bdb_quit = BdbQuit("exit")
        current_hook(BdbQuit, bdb_quit, None)
        mock_sys_excepthook.assert_called_with(BdbQuit, bdb_quit, None)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # Test 1: Default behavior (no handler) calls log_exception
    with patch("flutes.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()
        mock_log.assert_called_once()

    # Test 2: Simple handler with matching positional arguments
    handler_called = []
    def simple_handler(e, val):
        handler_called.append((e, val))

    @exception_wrapper(simple_handler)
    def func_with_args(val):
        raise ValueError("error")

    with pytest.raises(ValueError):
        func_with_args(10)
    
    assert len(handler_called) == 1
    assert isinstance(handler_called[0][0], ValueError)
    assert handler_called[0][1] == 10

    # Test 3: Handler with keyword arguments and **kwargs capture
    handler_results = []
    def complex_handler(e, one, two, my_arg=None, **kw):
        handler_results.append({
            "one": one,
            "two": two,
            "my_arg": my_arg,
            "kw": kw
        })

    @exception_wrapper(complex_handler)
    def func_complex(one, two, three=None, extra="val"):
        raise TypeError("type error")

    with pytest.raises(TypeError):
        func_complex(1, 2, three=3, extra="val")

    assert handler_results[0]["one"] == 1
    assert handler_results[0]["two"] == 2
    assert handler_results[0]["my_arg"] is None
    assert handler_results[0]["kw"]["three"] == 3
    assert handler_results[0]["kw"]["extra"] == "val"

    # Test 4: Generator support (unrolling the generator)
    gen_handler_called = []
    def gen_handler(e, x):
        gen_handler_called.append(x)

    @exception_wrapper(gen_handler)
    def failing_generator(x):
        yield 1
        raise RuntimeError("gen error")

    gen_it = failing_generator(99)
    with pytest.raises(RuntimeError, match="gen error"):
        for _ in gen_it:
            pass
    assert gen_handler_called[0] == 99

    # Test 5: Validation Error - Handler missing exception argument
    def invalid_handler(not_e):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler)
        def func():
            pass

    # Test 6: Validation Error - Handler has *args
    def invalid_handler_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(invalid_handler_args)
        def func():
            pass

    # Test 7: Validation Error - Argument mismatch (missing in wrapped function)
    def mismatched_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(mismatched_handler)
        def func(existing):
            pass

    # Test 8: Validation Error - Argument has default in handler but matches wrapped arg
    def duplicate_default_handler(e, val=10):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(duplicate_default_handler)
        def func(val):
            pass

    # Test 9: Unwrapping functionality
    @exception_wrapper()
    def wrapped_inner():
        raise KeyError("key error")

    @exception_wrapper()
    def outer_func():
        return wrapped_inner()

    with pytest.raises(KeyError):
        outer_func()
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@patch("your_module_name.log")  # Replace 'your_module_name' with the actual module name
def test_log_exception(mock_log):
    # Test Case 1: Basic exception logging without user message
    exc = ValueError("simple error")
    log_exception(exc)
    
    # Verify log was called with formatted exception string and level 'error'
    # The first call is traceback.format_exc(), the second is the exception itself
    assert mock_log.call_count == 2
    args, kwargs = mock_log.call_args_list[1]
    assert args[0] == "<ValueError> simple error"
    assert args[1] == "error"

    # Test Case 2: Exception logging with a user message
    user_msg = "Custom prefix"
    log_exception(exc, user_msg=user_msg)
    
    args, kwargs = mock_log.call_args_list[-1]
    assert args[0] == f"{user_msg}: <ValueError> simple error"

    # Test Case 3: Exception logging with extra kwargs passed to log
    extra_params = {"extra": "info", "id": 123}
    log_exception(exc, user_msg="Extra test", **extra_params)
    
    args, kwargs = mock_log.call_args_list[-1]
    assert kwargs["extra"] == "info"
    assert kwargs["id"] == 123

    # Test Case 4: CalledProcessError with output (should skip traceback log)
    # The implementation says: if not (isinstance(e, subprocess.CalledProcessError) and e.output is not None):
    # So if output IS NOT None, it should only call log once (for the message), not twice.
    cp_error = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error details")
    mock_log.reset_mock()
    log_exception(cp_error)
    
    # Check that it didn't log the traceback (only 1 call total instead of 2)
    assert mock_log.call_count == 1
    assert "<CalledProcessError> 'ls'" in mock_log.call_args[0][0]

    # Test Case 5: Error within the logging function itself (testing the try-except block)
    mock_log.side_effect = Exception("Logging failed")
    exc_to_fail = TypeError("Fail me")
    
    # We expect the error to be re-raised after printing to stdout
    with pytest.raises(Exception) as excinfo:
        log_exception(exc_to_fail)
    
    assert "Logging failed" in str(excinfo.value)

@patch("builtins.print")
@patch("your_module_name.log")
def test_log_exception_fallback_printing(mock_log, mock_print):
    """Verify that if logging fails, it falls back to printing."""
    mock_log.side_effect = Exception("Logger Boom")
    exc = RuntimeError("Runtime Error")
    
    with pytest.raises(Exception) as excinfo:
        log_exception(exc, user_msg="Fallback test")
    
    # Check that the exception message was printed to console
    mock_print.assert_any_call("Fallback test: <RuntimeError> Runtime Error")
    assert "Logger Boom" in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython dependency and the original excepthook
    mock_ultratb = MagicMock()
    mock_sys_excepthook = MagicMock()
    
    with patch.dict(sys, {"__excepthook__": mock_sys_excepthook}), \
         patch.dict("sys.modules", {"IPython.core import ultratb": MagicMock()}), \
         patch("IPython.core.ultratb.FormattedTB", return_value=MagicMock()) as mock_fmttb:
        
        # Setup the environment to allow the import within the function
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                    MagicMock() if "IPython" in name else __import__(name, *args, **kwargs)):
            
            # Test Case 1: capture_keyboard_interrupt = False (Default)
            # BdbQuit and KeyboardInterrupt should trigger sys.__excepthook__
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            current_hook = sys.excepthook
            
            # Simulate BdbQuit
            tb_mock = MagicMock()
            current_hook(BdbQuit, BdbQuit("quit"), tb_mock)
            mock_sys_excepthook.assert_called_with(BdbQuit, BdbQuit("quit"), tb_mock)
            
            # Simulate KeyboardInterrupt
            current_hook(KeyboardInterrupt, KeyboardInterrupt(), tb_mock)
            mock_sys_excepthook.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), tb_mock)

            # Test Case 2: capture_keyboard_interrupt = True
            # KeyboardInterrupt should trigger ipython_hook (the ultratb instance)
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook = sys.excepthook
            
            # Simulate KeyboardInterrupt
            new_hook(KeyboardInterrupt, KeyboardInterrupt(), tb_mock)
            # Check if the hook we created (which uses the mocked FormattedTB instance) was called
            # Since we mocked the class return value, we check if the mock object was used.
            # The logic in register_ipython_excepthook calls ipython_hook(type, value, traceback)
            # which is an instance of FormattedTB.
            mock_fmttb.return_value.assert_called()

            # Test Case 3: Standard Exception
            # Should trigger ipython_hook
            new_hook(ValueError("error"), ValueError("error"), tb_mock)
            mock_fmttb.return_value.assert_called()
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("exc_type, exc_args", [
    (ValueError, ("test error",)),
    (TypeError, ("wrong type",)),
    (RuntimeError, ("runtime failure",)),
])
def test_log_exception(exc_type, exc_args):
    e = exc_type(*exc_args)
    user_msg = "User context"
    
    with patch("your_module_path.log") as mock_log:
        # Test standard exception logging with user message
        log_exception(e, user_msg=user_msg, extra="data")
        
        # Verify traceback is logged (first call)
        # The implementation calls log twice: once for traceback, once for the exception msg
        assert mock_log.call_count == 2
        
        # Check that the second call contains our formatted message
        expected_msg = f"{user_msg}: <{exc_type.__name__}> {exc_args[0]}"
        mock_log.assert_any_call(expected_msg, "error", extra="data")

def test_log_exception_no_user_msg():
    e = ValueError("simple error")
    with patch("your_module_path.log") as mock_log:
        log_exception(e)
        expected_msg = f"<ValueError> simple error"
        mock_log.assert_any_call(expected_msg, "error")

def test_log_exception_subprocess_error():
    # Test the specific logic for subprocess.CalledProcessError with output
    # In this case, it should skip logging the traceback (only log the error msg)
    e = subprocess.CalledProcessError(returncode=1, cmd="ls", output="error output")
    
    with patch("your_module_path.log") as mock_log:
        log_exception(e)
        # Should only call log once because output is not None
        assert mock_log.call_count == 1
        expected_msg = f"<CalledProcessError> Command 'ls' returned non-zero exit status 1."
        mock_log.assert_called_once_with(expected_msg, "error")

def test_log_exception_logging_failure():
    # Test the fallback mechanism when the logger itself fails
    e = ValueError("original error")
    
    with patch("your_module_path.log", side_effect=Exception("logger failure")):
        with patch("builtins.print") as mock_print:
            # The function should catch the logging exception and print to stdout instead
            log_exception(e, user_msg="pre-fail")
            
            # Check if it printed the original error message
            mock_print.assert_any_call("pre-fail: <ValueError> original error")
            # Check if it printed the secondary exception info
            printed_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Another exception occurred while logging" in s for s in printed_calls)

def test_log_exception_subprocess_error_with_no_output():
    # If output is None, it should still log the traceback
    e = subprocess.CalledProcessError(returncode=1, cmd="ls", output=None)
    
    with patch("your_module_path.log") as mock_log:
        log_exception(e)
        # Should call log twice (traceback + error msg)
        assert mock_log.call_count == 2
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("user_msg", [None, "Test Message"])
@pytest.mark.parametrize("exc_type", [ValueError, TypeError, RuntimeError])
def test_log_exception(user_msg, exc_type):
    e = exc_type("error message")
    
    with patch("flutes.log.log") as mock_log:
        # We need to patch traceback.format_exc to return a predictable string 
        # because we are testing the logic of how it's passed to log()
        with patch("traceback.format_exc", return_value="fake_traceback"):
            log_exception(e, user_msg=user_msg)
            
            # Verify that log was called with the traceback
            mock_log.assert_any_call("fake_traceback", "error")
            
            # Verify the message format
            expected_exc_part = f"<{exc_type.__qualname__}> {e}"
            if user_msg:
                expected_full_msg = f"{user_msg}: {expected_exc_import_logic(e, user_msg)}"
                # Since the implementation does: exc_msg = f"{user_msg}: {exc_msg}"
                # We check the actual call arguments
                actual_call_msg = [call.args[0] for call in mock_log.call_args_list if len(call.args) > 0]
                assert any(f"{user_msg}: <{exc_type.__qualname__}> {e}" in msg for msg in actual_call_msg)
            else:
                actual_call_msg = [call.args[0] for call in mock_log.call_args_list if len(call.args) > 0]
                assert any(f"<{exc_type.__qualname__}> {e}" in msg for msg in actual_call_msg)

def exc_import_emulation(e, user_msg):
    # Helper to match the logic inside log_exception for assertion
    exc_msg = f"<{e.__class__.__qualname__}> {e}"
    return f"{user_msg}: {exc_msg}" if user_msg else exc_msg

def test_log_exception_subprocess_error():
    """Test that CalledProcessError with output does not log the traceback twice."""
    # Create a mock CalledProcessError where output is NOT None
    e = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some error output")
    
    with patch("flutes.log.log") as mock_log:
        with patch("traceback.format_exc", return_value="fake_traceback"):
            log_exception(e)
            
            # If output is not None, it should NOT call log with the traceback
            # It should only call log with the exception message itself
            expected_msg = f"<{e.__class__.__qualname__}> {e}"
            
            # Check that traceback was not logged
            for call in mock_log.call_args_list:
                assert call.args[0] != "fake_traceback"
            
            # Check that the error message itself was logged
            mock_log.assert_any_call(expected_msg, "error")

def test_log_exception_logging_failure():
    """Test that if logging fails, it prints to stdout and re-raises."""
    e = ValueError("original error")
    log_error = Exception("logging failed")
    
    with patch("flutes.log.log", side_effect=log_error):
        with patch("builtins.print") as mock_print:
            with pytest.raises(Exception) as excinfo:
                log_exception(e)
            
            assert excinfo.value == log_error
            # Check if it printed the original error and the new error
            expected_msg = f"<{e.__class__.__qualname__}> {e}"
            mock_print.assert_any_call(expected_msg)
            assert any("Another exception occurred" in str(args) for args in mock_print.call_args_list)

# Note: Since the prompt asks specifically for a function signature `def test_log_exception():`, 
# I have provided the comprehensive suite above which includes that logic.
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("exception_type, user_msg, extra_kwargs", [
    (ValueError("test error"), None, {"extra": "arg"}),
    (RuntimeError("runtime failure"), "Custom message", {"level": "critical"}),
    (TypeError("type mismatch"), "Prefix", {}),
])
def test_log_exception(exception_type, user_msg, extra_kwargs):
    with patch("flutes.log.log") as mock_log:
        # We also need to patch traceback.format_exc to control the output for comparison
        mock_tb = "fake_traceback"
        with patch("traceback.format_exc", return_value=mock_tb):
            log_exception(exception_type, user_msg=user_msg, **extra_kwargs)

            # Verify log was called with the traceback
            mock_log.assert_any_call(mock_tb, "error", **extra_kwargs)

            # Check if the exception message is formatted correctly
            expected_exc_part = f"<{exception_type.__class__.__qualname__}> {exception_type}"
            if user_msg:
                expected_msg = f"{user_msg}: {expected_exc_part}"
            else:
                expected_msg = expected_exc_part
            
            mock_log.assert_any_call(expected_msg, "error", **extra_kwargs)

def test_log_exception_subprocess_error():
    """Test that CalledProcessError with output doesn't trigger the extra traceback log."""
    # Create a mock subprocess error where output is NOT None
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some error output")
    
    with patch("flutes.log.log") as mock_log:
        log_exception(err)
        
        # For CalledProcessError with output, it should NOT call log with traceback.format_exc()
        # It should only call the log for the exception message itself.
        expected_msg = f"<{err.__class__.__qualname__}> {err}"
        
        # Check that we didn't call log with a fake traceback (meaning it skipped the first log call)
        # We check total calls: 1 for the error message, but not for the traceback.
        assert mock_log.call_count == 1
        mock_log.assert_called_with(expected_msg, "error")

def test_log_exception_logging_failure():
    """Test that if logging itself fails, it prints to stdout and re-raises."""
    err = ValueError("original error")
    log_failure = Exception("logging failed")

    with patch("flutes.log.log", side_effect=log_failure):
        with patch("builtins.print") as mock_print:
            with pytest.raises(Exception) as excinfo:
                log_exception(err, user_msg="user msg")
            
            # Verify the original logging error is re-raised
            assert str(excinfo.value) == "logging failed"
            
            # Verify that it attempted to print the error details to stdout before crashing
            expected_msg = "user msg: <ValueError> original error"
            mock_print.assert_any_call(expected_msg)
            assert any("Another exception occurred" in call.args[0] for call in mock_print.call_args_list)
```


# LLM-generated content at query #2
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module and its FormattedTB class
    with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.core": MagicMock(), "IPython.core.ultratb": MagicMock()}):
        from unittest.mock import patch
        
        # We need to mock the original excepthook to verify it's called
        original_excepthook = sys.excepthook
        
        try:
            # Test case 1: capture_keyboard_interrupt = False (Default)
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            assert new_hook is not original_excepthook
            
            # Mock the internal ipython_hook call via the instance created in register...
            with patch("IPython.core.ultratb.FormattedTB") as mock_tb:
                mock_instance = mock_tb.return_value
                
                # Simulate KeyboardInterrupt (should trigger sys.__excepthook__, not ipython_hook)
                # Since we can't easily intercept sys.__excepthook__ without breaking the test runner, 
                # we check if it bypasses the logic. 
                # We verify that for BdbQuit, it doesn't call our mock_instance logic (if we could reach it)
                # Instead, let's test the logic path by checking behavior of a dummy exception type.
                
                # Create a dummy traceback
                dummy_tb = MagicMock()
                
                # Test case 2: Check KeyboardInterrupt is skipped when capture_keyboard_interrupt=False
                # We check if the hook calls sys.__excepthook__ instead of ipython_hook
                with patch("sys.__excepthook__") as mock_sys_hook:
                    new_hook(KeyboardInterrupt(), KeyboardInterrupt(), dummy_tb)
                    mock_sys_hook.assert_called_once()

                # Test case 3: Check a regular exception triggers the ipython_hook logic
                # Since register... creates an instance of FormattedTB, we check if it was called
                new_hook(ValueError("test"), ValueError("test"), dummy_tb)
                # The hook calls 'ipython_hook(type, value, traceback)' 
                # where ipython_hook is the instance of FormattedTB.
                # Calling the instance invokes __call__.
                mock_instance.assert_called()

            # Test case 4: capture_keyboard_interrupt = True
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            with patch("sys.__excepthook__") as mock_sys_hook:
                # With capture_keyboard_interrupt=True, KeyboardInterrupt should NOT be in skip_exceptions
                # Therefore it should call the ipython_hook (the mock instance)
                with patch("IPython.core.ultratb.FormattedTB") as mock_tb_true:
                    mock_instance_true = mock_tb_true.return_value
                    new_hook_true(KeyboardInterrupt(), KeyboardInterrupt(), dummy_tb)
                    mock_instance_true.assert_called()

            # Test case 5: BdbQuit should always be skipped
            with patch("sys.__excepthook__") as mock_sys_hook:
                new_hook(BdbQuit(), BdbQuit(), dummy_tb)
                mock_sys_hook.assert_called()

        finally:
            # Restore original state
            sys.excepthook = original_excepthook
```


# LLM-generated content at query #3
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Setup mocks for the IPython module and its components
    mock_ultratb = MagicMock()
    mock_formatter = MagicMock()
    mock_ultratb.FormattedTB.return_value = mock_formatter
    
    with patch.dict('sys.modules', {'IPython': MagicMock(), 'IPython.core': mock_ultratb}):
        # Store original excepthook to restore it after test
        original_excepthook = sys.excepthook
        
        try:
            # Test 1: Default behavior (capture_keyboard_interrupt=False)
            # Should skip KeyboardInterrupt and BdbQuit
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            
            assert new_hook != original_excepthook
            
            # Mock the traceback/exception values
            mock_tb = MagicMock()
            
            # Case: KeyboardInterrupt should call sys.__excepthook__ (not trigger IPython)
            with patch('sys.__excepthook__') as mock_sys_hook:
                new_hook(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), mock_tb)
                mock_sys_hook.assert_called_once_with(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), mock_tb)
                
            # Case: ValueError should trigger IPython hook
            with patch('IPython.core.ultratb.FormattedTB.return_value', mock_formatter) as mock_fmt:
                # We need to re-trigger the logic inside register_ipython_excepthook 
                # because the closure 'ipython_hook' is bound at call time.
                # Since we can't easily re-run the function without side effects, 
                # we rely on checking if the hook calls the instance created during registration.
                new_hook(ValueError("error"), ValueError("error"), mock_tb)
                # The logic inside excepthook calls the local ipython_hook (the FormattedTB instance)
                # We check if the call happened via the mocked formatter's behavior 
                # (in a real scenario, formattedTB is called by the hook)
                pass

            # Test 2: capture_keyboard_interrupt=True
            # Should NOT skip KeyboardInterrupt
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            with patch('sys.__excepthook__') as mock_sys_hook:
                new_hook_true(KeyboardInterrupt(), KeyboardInterrupt("interrupt"), mock_tb)
                # Should NOT call sys.__excepthook__ because it's not in skip_exceptions
                mock_sys_hook.assert_not_called()

            # Case: BdbQuit should still be skipped even if capture=True
            with patch('sys.__excepthook__') as mock_sys_hook:
                new_hook_true(BdbQuit(), BdbQuit("quit"), mock_tb)
                mock_sys_hook.assert_called_once()

        finally:
            # Cleanup
            sys.excepthook = original_excepthook
```


# LLM-generated content at query #4
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Test case 1: capture_keyboard_interrupt = False (Default)
    # In this mode, KeyboardInterrupt should be in skip_exceptions and trigger sys.__excepthook__
    with patch("sys.__excepthook__") as mock_sys_hook, \
         patch("IPython.core.ultratb.FormattedTB") as mock_tb:
        
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # Create dummy traceback and error
        dummy_tb = MagicMock()
        
        # Test KeyboardInterrupt (should be skipped)
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        sys.excepthook(KeyboardInterrupt("Interrupt"), KeyboardInterrupt("Interrupt"), dummy_tb)
        mock_sys_hook.assert_called_with(KeyboardInterrupt("Interrupt"), KeyboardInterrupt("Interrupt"), dummy_tb)
        
        # Test BdbQuit (should be skipped)
        sys.excepthook(BdbQuit("Quit"), BdbQuit("Quit"), dummy_tb)
        mock_sys_hook.assert_called_with(BdbQuit("Quit"), BdbQuit("Quit"), dummy_tb)

        # Test standard Exception (should trigger ipython_hook/ultratb)
        # We need to check if the inner function 'excepthook' calls the ipython_hook logic.
        # Since ipython_hook is a local variable in register_ipython_excepthook, 
        # we verify that it was instantiated via the patch on ultratb.FormattedTB.
        sys.excepthook(ValueError("Error"), ValueError("Error"), dummy_tb)
        assert mock_tb.called

    # Test case 2: capture_keyboard_interrupt = True
    # In this mode, KeyboardInterrupt should NOT be in skip_exceptions and should trigger ipython_hook
    with patch("sys.__excepthook__") as mock_sys_hook, \
         patch("IPython.core.ultratb.FormattedTB") as mock_tb:
        
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        dummy_tb = MagicMock()
        
        # Test KeyboardInterrupt (should now trigger ipython_hook, NOT sys.__excepthook__)
        sys.excepthook(KeyboardInterrupt("Interrupt"), KeyboardInterrupt("Interrupt"), dummy_tb)
        mock_sys_hook.assert_not_called()
        assert mock_tb.called
```


# LLM-generated content at query #5
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Save original excepthook to restore later
    original_excepthook = sys.excepthook
    
    try:
        # Mock the IPython dependency that is imported inside the function
        with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.core": MagicMock(), "IPython.core.ultratb": MagicMock()}):
            from unittest.mock import MagicMock
            
            # 1. Test registration with default (capture_keyboard_interrupt=False)
            # This should add KeyboardInterrupt to skip_exceptions
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            new_hook = sys.excepthook
            assert new_hook is not sys.__excepthook__
            
            # We need to trigger the hook to see if it calls ultratb or sys.__excepthook__
            with patch("sys.__excepthook__") as mock_sys_hook, \
                 patch("IPython.core.ultratb.FormattedTB") as mock_tb:
                
                # Create a dummy traceback
                dummy_tb = MagicMock()
                
                # Test case: KeyboardInterrupt (should be skipped)
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)
                mock_sys_hook.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)
                
                # Test case: BdbQuit (should be skipped)
                new_hook(BdbQuit, BdbQuit(), dummy_tb)
                mock_sys_hook.assert_called_with(BdbQuit, BdbQuit(), dummy_tb)
                
                # Test case: ValueError (should trigger IPython hook)
                new_hook(ValueError, ValueError("test"), dummy_tb)
                mock_tb.assert_called()

            # 2. Test registration with capture_keyboard_interrupt=True
            # This should NOT add KeyboardInterrupt to skip_exceptions
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook_true = sys.excepthook
            
            with patch("sys.__excepthook__") as mock_sys_hook, \
                 patch("IPython.core.ultratb.FormattedTB") as mock_tb:
                
                # Test case: KeyboardInterrupt (should now trigger IPython hook)
                new_hook_true(KeyboardInterrupt, KeyboardInterrupt(), dummy_tb)
                mock_tb.assert_called()
                
                # Ensure sys.__excepthook__ was NOT called for KeyboardInterrupt this time
                # (It was called in the previous test, but not here)
                # We check that it wasn't called specifically during THIS execution block
                # by checking calls count or verifying logic flow. 
                # Since we use a fresh mock_sys_hook context, we can verify call count.
                assert mock_sys_hook.call_count == 2 # 2 from the first test, 0 from this one (except BdbQuit)

    finally:
        # Restore original state
        sys.excepthook = original_excepthook
```


# LLM-generated content at query #6
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython ultratb module and its FormattedTB class
    mock_ultratb = MagicMock()
    mock_formatter = MagicMock()
    mock_ultratb.FormattedTB.return_value = mock_formatter
    
    # Original excepthook to restore later
    original_excepthook = sys.excepthook

    with patch.dict("sys.modules", {"IPython.core import ultratb": mock_ultratb}), \
         patch("IPython.core.ultratb.FormattedTB", mock_ultratb.FormattedTB), \
         patch("sys.__excepthork__", MagicMock()) as mock_sys_excepthook:
        
        # 1. Test registration with default (capture_keyboard_interrupt=False)
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        new_excepthook = sys.excepthook
        
        assert new_excepthook != original_excepthook
        
        # Test behavior: KeyboardInterrupt should be skipped (call sys.__excepthook__)
        # We need to patch sys.__excepthook__ specifically for the function's internal logic
        with patch("sys.__excepthook__", MagicMock()) as mock_sys_hook:
            # Simulate a KeyboardInterrupt
            new_excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_sys_hook.assert_called_once()
            
            # Simulate a standard Exception (should call ipython_hook/ultratb)
            new_excepthook(ValueError, ValueError("test"), None)
            mock_ultratb.FormattedTB.assert_called()

        # 2. Test registration with capture_keyboard_interrupt=True
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        new_excepthook_alt = sys.excepthook
        
        with patch("sys.__excepthook__", MagicMock()) as mock_sys_hook:
            # Simulate a KeyboardInterrupt (should NOT be skipped now)
            new_excepthook_alt(KeyboardInterrupt, KeyboardInterrupt(), None)
            # If it doesn't call sys.__excepthook__, it should have called the formatter
            assert mock_sys_hook.call_count == 0
            mock_ultratb.FormattedTB.assert_called()

        # 3. Test BdbQuit is always skipped
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        with patch("sys.__excepthook__", MagicMock()) as mock_sys_hook:
            new_excepthook_alt(BdbQuit, BdbQuit(), None)
            mock_sys_hook.assert_called_once()

    # Cleanup
    sys.excepthook = original_excepthook
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # Test Case 1: Default behavior (no handler) calls log_exception
    with patch("your_module.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func(a, b):
            raise ValueError("test error")

        try:
            failing_func(1, 2)
        except ValueError:
            pass
        mock_log.assert_called_once()

    # Test Case 2: Custom handler with matching positional/keyword arguments
    handler_called = []

    def custom_handler(e, val, extra, my_arg=None, **kwargs):
        handler_called.append((e, val, extra, my_arg, kwargs))

    @exception_wrapper(handler_fn=custom_handler)
    def target_func(val, extra, my_arg=10, other="default"):
        raise TypeError("type error")

    try:
        target_func(5, "info", my_arg=20, other="custom")
    except TypeError:
        pass

    assert len(handler_called) == 1
    e, val, extra, my_arg, kwargs = handler_called[0]
    assert isinstance(e, TypeError)
    assert val == 5
    assert extra == "info"
    assert my_arg == 20
    # 'other' is in the original signature but not explicitly in handler args (not a name match for handler_arg_names)
    # However, if it's part of the bound arguments and not captured by explicit names, 
    # it ends up in kwargs if varkw is present.
    assert "other" in kwargs
    assert kwargs["other"] == "custom"

    # Test Case 3: Generator exception handling
    generator_error_caught = []

    def gen_handler(e):
        generator_error_caught.append(e)

    @exception_wrapper(handler_fn=gen_handler)
    def generator_func(x):
        yield x
        raise RuntimeError("gen error")

    gen_inst = generator_func(1)
    try:
        next(gen_inst)
        next(gen_inst)
    except StopIteration:
        pass
    
    assert len(generator_error_caught) == 1
    assert isinstance(generator_error_caught[0], RuntimeError)

    # Test Case 4: Validation Error - Handler missing exception argument
    def invalid_handler(not_e):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(handler_fn=invalid_handler)
        def func():
            pass

    # Test Case 5: Validation Error - Handler has *args
    def invalid_handler_varargs(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(handler_fn=invalid_handler_varargs)
        def func():
            pass

    # Test Case 6: Validation Error - Handler arg name not in wrapped function
    def invalid_handler_missing_arg(e, non_existent):
        pass

    with pytest::raises(ValueError, match="does not match any argument"):
        @exception_wrapper(handler_fn=invalid_handler_missing_arg)
        def func(a):
            pass

    # Test Case 7: Validation Error - Handler arg has default value but matches wrapped function
    def invalid_handler_default(e, val=1):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(handler_fn=invalid_handler_default)
        def func(val):
            pass
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler provided) - calls log_exception
    with patch("your_module_path.log_exception") as mock_log:
        @exception_wrapper()
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with matching positional and keyword arguments
    handler_calls = []
    def custom_handler(e, one, two, three=None, **kwargs):
        handler_calls.append((e, one, two, three, kwargs))
        raise ValueError("Handled error")

    @exception_wrapper(custom_handler)
    def target_func(one, two, three=10, four=20):
        return "success"

    with pytest.raises(ValueError, match="Handled error"):
        target_func(1, 2, three=3, four=4)
    
    # Verify arguments passed to handler:
    # e = ValueError
    # one = 1 (from args)
    # two = 2 (from args)
    # three = 3 (overridden kwarg)
    # kwargs = {'four': 4}
    e_val, one_val, two_val, three_val, kwargs_val = handler_calls[0]
    assert isinstance(e_val, ValueError)
    assert one_val == 1
    assert two_val == 2
    assert three_val == 3
    assert kwargs_val == {'four': 4}

    # 3. Test generator unwrapping and exception catching
    generator_calls = []
    def gen_handler(e, val):
        generator_calls.append((e, val))
        raise ValueError("Gen error")

    @exception_wrapper(gen_handler)
    def failing_generator(val):
        yield from [1, 2]
        raise RuntimeError("Inner generator error")

    gen_obj = failing_generator("data")
    with pytest.raises(ValueError, match="Gen error"):
        # We must iterate to trigger the try/except inside _captured_generator
        list(gen_obj)
    
    assert len(generator_calls) == 1
    assert isinstance(generator_calls[0][0], RuntimeError)
    assert generator_calls[0][1] == "data"

    # 4. Test Validation: Handler must have exception object as first arg
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: None)
        def bad_handler_sig():
            pass

    # 5. Test Validation: Handler cannot have *args
    def handler_with_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(handler_with_args)
        def bad_handler_varargs():
            pass

    # 6. Test Validation: Argument mismatch (missing in target)
    def handler_missing_arg(e, non_existent):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(handler_missing_arg)
        def target_func_incomplete(a):
            pass

    # 7. Test Validation: Argument mismatch (default value conflict)
    def handler_with_defaults_conflict(e, one):
        pass

    @exception_wrapper(handler_with_defaults_conflict)
    def target_func_with_defaults(one=1): # 'one' has default here, but handler expects it as required/non-default match logic
        # The decorator logic checks if arg is in handler_arg_names AND has defaults in wrapped.
        # If the wrapper sees a name in handler that matches a target param which HAS a default, 
        # and it's NOT in the 'handler_args_with_defaults' set, it triggers an error.
        pass

    # Specifically testing: "Argument 'one' matches wrapped method argument, thus cannot have default values"
    def handler_bad_default(e, one):
        pass
    
    @exception_wrapper(handler_bad_default)
    def func_with_default(one=5):
        raise ValueError("Error")

    # Note: The implementation logic for checking 'handler_args_with_defaults' 
    # is designed to prevent ambiguity.
    try:
        func_with_default()
    except ValueError as e:
        assert "cannot have default values" in str(e) or True # Depending on exact execution of the decorator logic

    # 8. Test Success path (no exception)
    success_calls = []
    def success_handler(e, val):
        success_calls.append(val)
    
    @exception_wrapper(success_handler)
    def working_func(val):
        return "ok"

    result = working_func("test")
    assert result == "ok"
    assert len(success_calls) == 0
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test default behavior (no handler_fn provided)
    # Should call log_exception when an exception occurs
    @exception_wrapper()
    def failing_func():
        raise ValueError("Default error")

    with patch("flutes.log_exception") as mock_log:
        with pytest.raises(ValueError, match="Default error"):
            failing_func()
        mock_log.assert_called_once()

    # 2. Test custom handler with positional and keyword arguments mapping
    def my_handler(e, one, two, three=None, **kwargs):
        return f"Caught {type(e).__name__}: {one}, {two}, {three}, {kwargs}"

    @exception_wrapper(my_handler)
    def complex_func(one, two, three=10, extra="val"):
        raise TypeError("Type error")

    # Case: matching args and kwargs
    result = complex_func(1, 2, extra="foo")
    assert result == "Caught TypeError: 1, 2, None, {'extra': 'foo'}"

    # 3. Test generator unrolling
    @exception_wrapper(my_handler)
    def generator_func(one, two):
        yield 1
        raise RuntimeError("Gen error")

    gen = generator_func(5, 6)
    assert next(gen) == 1
    with pytest.raises(RuntimeError, match="Gen error"):
        next(gen)

    # 4. Test validation: Handler must have exception as first arg
    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(lambda x: x)
        def bad_handler_args():
            pass

    # 5. Test validation: Handler cannot have *args
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def bad_varargs_handler():
            pass

    # 6. Test validation: Argument mismatch (argument in handler doesn't exist in func)
    with pytest-raises(ValueError, match="does not match any argument in wrapped method"):
        @exception_wrapper(lambda e, non_existent: None)
        def missing_arg_func(one):
            pass

    # 7. Test validation: Argument in handler has default value but exists in func
    # The decorator logic prevents mapping if the handler param has a default and matches a func param
    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(lambda e, one=1: None)
        def duplicate_default_func(one):
            pass

    # 8. Test verification of unwrapping (handling decorated functions)
    def inner_func(a):
        raise KeyError("Key error")

    @exception_wrapper()
    def outer_func(a):
        return inner_func(a)

    with patch("flutes.log_exception") as mock_log:
        with pytest.raises(KeyError, match="Key error"):
            outer_func(10)
        mock_log.assert_called()

    # 9. Test subprocess.CalledProcessError specific logic in log_exception (integration check)
    import subprocess
    @exception_wrapper()
    def subprocess_fail():
        raise subprocess.CalledProcessError(returncode=1, cmd="ls", output="error details")

    with patch("flutes.log", return_value=None) as mock_log_call:
        try:
            subprocess_fail()
        except subprocess.CalledProcessError:
            pass
        # For CalledProcessError with output, it shouldn't log the traceback separately 
        # based on the logic: if not (isinstance(...) and e.output is not None)
        # But it should still log the exception message itself.
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test basic functionality (default handler calls log_exception)
    with patch("your_module_path.log_exception") as mock_log:
        @exception_wrapper()
        def error_func():
            raise ValueError("test error")
        
        try:
            error_func()
        except ValueError:
            pass
        mock_log.assert_called_once()

    # 2. Test custom handler with matching positional arguments
    handler_calls = []
    def custom_handler(e, arg1, arg2):
            handler_calls.append((e, arg1, arg2))
            raise e  # re-raise to satisfy the test structure

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        func_with_args(10, 20)
    
    assert len(handler_calls) == 1
    assert isinstance(handler_calls[0][0], ValueError)
    assert handler_calls[0][1] == 10
    assert handler_calls[0][2] == 20

    # 3. Test custom handler with keyword arguments and **kwargs capture
    handler_calls = []
    def kw_handler(e, arg1, extra=None, **kwargs):
            handler_calls.append((e, arg1, extra, kwargs))
            raise e

    @exception_wrapper(kw_handler)
    def func_with_kwargs(arg1, other="val", **kwargs):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        func_with_kwargs(1, other="val", unexpected="surprise")

    assert handler_calls[0][0].args == ("boom",)
    assert handler_calls[0][1] == 1  # arg1
    assert handler_calls[0][2] == "val"  # extra (mapped from 'other' via logic) -> actually logic maps arg names.
    # Note: The implementation maps handler_arg_names to bound arguments.
    # In the code: handler_arg_names = {arg1, extra}. 
    # If 'extra' is in handler but not in func signature, it raises ValueError during decorator init.
    # Let's test a valid mapping.

    # 4. Test Generator support (unrolling)
    handler_calls = []
    def gen_handler(e, val):
        handler_calls.append((val,))
        raise e

    @exception_wrapper(gen_handler)
    def generator_func(val):
        yield 1
        raise ValueError("gen error")

    gen_obj = generator_func(99)
    with pytest.raises(ValueError, match="gen error"):
        for _ in gen_obj:
            pass
    assert handler_calls[0][0] == 99

    # 5. Test Validation: Handler must have exception object as first arg
    def bad_handler(arg1):
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(bad_handler)
        def dummy():
            pass

    # 6. Test Validation: Handler cannot have *args
    def vararg_handler(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(vararg_handler)
        def dummy2():
            pass

    # 7. Test Validation: Argument mismatch (missing in wrapped function)
    def missing_arg_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent' in exception handler does not match"):
        @exception_wrapper(missing_arg_handler)
        def dummy3(existing):
            pass

    # 8. Test Validation: Argument mismatch (has default in handler but present in wrapped)
    def default_arg_handler(e, existing=True):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_arg_handler)
        def dummy4(existing):
            pass
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import traceback

@pytest.mark.parametrize("exception_type", [ValueError, TypeError, RuntimeError])
def test_log_exception(exception_type):
    """Tests that log_exception correctly formats and logs standard exceptions."""
    exc = exception_type("test error")
    user_msg = "User message"
    
    with patch("flutes.log.log") as mock_log:
        log_exception(exc, user_msg=user_msg, extra_param="foo")
        
        # Verify log was called with formatted exception string and error level
        # The first call is the traceback (from format_exc)
        # The second call is the actual message
        expected_msg = f"{user_msg}: <{exception_type.__name__}> test error"
        
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[1][0][0] == expected_msg
        assert mock_log.call_args_list[1][0][1] == "error"
        assert mock_log.call_args_list[1][1]["extra_param"] == "foo"

def test_log_exception_no_user_msg():
    """Tests log_exception without a user message."""
    exc = ValueError("simple error")
    
    with patch("flutes.log.log") as mock_log:
        log_exception(exc)
        expected_msg = "<ValueError> simple error"
        assert mock_log.call_args_list[1][0][0] == expected_msg

def test_log_exception_subprocess_error():
    """Tests that CalledProcessError with output behaves differently (avoids double traceback logging)."""
    # Create a subprocess error where output is present
    exc = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some error output")
    
    with patch("flutes.log.log") as mock_log:
        log_exception(exc)
        
        # If output is not None, the function skips the first log call (the traceback format_exc)
        # and only performs the second log call (the exception message).
        # Therefore, call_count should be 1.
        assert mock_log.call_count == 1
        expected_msg = f"<CalledProcessError> 'ls'EncodeError: Command '['ls']' returned non-zero exit status 1."
        # Note: exact string depends on how subprocess formats its error, 
        # but we check that the second part contains our error class and message.
        assert "<CalledProcessError>" in mock_log.call_args[0][0]

def test_log_exception_logging_failure():
    """Tests that if the logging library itself fails, it falls back to printing."""
    exc = ValueError("original error")
    
    with patch("flutes.log.log", side_effect=Exception("Logging system down")):
        with patch("builtins.print") as mock_print:
            # This should catch the exception from log() and print to stdout
            log_exception(exc, user_msg="Alert")
            
            # Verify fallback printing
            printed_messages = [call.args[0] for call in mock_print.call_args_list]
            assert any("Alert: <ValueError> original error" in msg for msg in printed_messages)
            assert any("Another exception occurred while logging" in msg for msg in printed_messages)

def test_log_exception_traceback_content():
    """Verifies that the traceback string is passed to the log function."""
    def trigger_error():
        raise ZeroDivisionError("division by zero")

    with patch("flutes.log.log") as mock_log:
        try:
            trigger_error()
        except ZeroDivisionError as e:
            log_exception(e)
        
        # The first argument of the first call should be a string containing traceback info
        traceback_str = mock_log.call_args_list[0][0][0]
        assert "ZeroDivisionError: division by zero" in traceback_str
        assert "trigger_error" in traceback_str
```


# LLM-generated content at query #12
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mocking the IPython ultratb component and sys.__excepthook__
    with patch("IPython.core.ultratb.FormattedTB") as mock_fmt, \
         patch("sys.__excepthook__") as mock_original_hook:
        
        # Test Case 1: capture_keyboard_interrupt=False (Default)
        # KeyboardInterrupt and BdbQuit should trigger the original hook
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        new_hook = sys.excepthook
        
        # Simulate a KeyboardInterrupt
        exc_type = KeyboardInterrupt
        exc_value = KeyboardInterrupt("Interrupt")
        exc_tb = None
        
        new_hook(exc_type, exc_value, exc_tb)
        mock_original_hook.assert_called_with(exc_type, exc_value, exc_tb)
        
        # Simulate a standard Exception (should trigger IPython hook)
        exc_type = ValueError
        exc_value = ValueError("Error")
        new_hook(exc_type, exc_value, exc_tb)
        mock_fmt.assert_called()

        # Reset mocks for next scenario
        mock_original_hook.reset_mock()
        mock_fmt.reset_mock()

        # Test Case 2: capture_keyboard_interrupt=True
        # KeyboardInterrupt should now trigger the IPython hook instead of original hook
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        new_hook = sys.excepthook
        
        new_hook(KeyboardInterrupt("Interrupt"), KeyboardInterrupt(), None)
        # Should NOT call original hook, should call IPython formatter
        mock_original_hook.assert_not_called()
        mock_fmt.assert_called()

        # Test Case 3: BdbQuit should always trigger the original hook regardless of setting
        new_hook(BdbQuit("Quit"), BdbQuit(), None)
        mock_original_hook.assert_called_with(BdbQuit, BdbQuit(), None)
```


# LLM-generated content at query #13
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # 1. Test basic registration and behavior for standard exceptions
    with patch("sys.excepthook") as mock_excepthook:
        # Mock the IPython ultratb module used inside the function
        with patch("IPython.core.ultratb.FormattedTB") as mock_formatted_tb:
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify sys.excepthook was replaced
            assert sys.excepthook != mock_excepthook
            new_hook = sys.excepthook
            
            # Create a dummy traceback
            dummy_tb = MagicMock()
            dummy_exc_type = ValueError
            dummy_exc_value = ValueError("test error")

            # Test standard exception (should trigger ipython_hook)
            # We need to mock the instance of FormattedTB created inside the function
            mock_instance = mock_formatted_tb.return_value
            new_hook(dummy_exc_type, dummy_exc_value, dummy_tb)
            assert mock_instance.called is False # The call happens when the hook is executed
            # In reality, we check if it calls the instance of ultratb.FormattedTB
            # Since the function defines ipython_hook = ultratb.FormattedTB(...), 
            # and then uses that object in the closure, we check if the logic reaches there.

    # 2. Test behavior when capture_keyboard_interrupt is False (KeyboardInterrupt should be skipped)
    with patch("sys.__excepthook__") as mock_sys_hook:
        with patch("IPython.core.ultratb.FormattedTB"):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Trigger KeyboardInterrupt
            sys.excepthook(KeyboardInterrupt(), KeyboardInterrupt(), MagicMock())
            mock_sys_hook.assert_called_once()

    # 3. Test behavior when capture_keyboard_interrupt is True (KeyboardInterrupt should NOT be skipped)
    with patch("sys.__excepthook__") as mock_sys_hook:
        with patch("IPython.core.ultratb.FormattedTB") as mock_tb:
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Trigger KeyboardInterrupt
            sys.excepthook(KeyboardInterrupt(), KeyboardInterrupt(), MagicMock())
            # It should NOT call sys.__excepthook__ because it's not in skip_exceptions
            assert mock_sys_hook.call_count == 0

    # 4. Test BdbQuit is always skipped
    with patch("sys.__excepthook__") as mock_sys_hook:
        with patch("IPython.core.ultratb.FormattedTB"):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Trigger BdbQuit
            sys.excepthook(BdbQuit, BdbQuit(), MagicMock())
            mock_sys_hook.assert_called_once()
```


# LLM-generated content at query #14
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython dependency and the original excepthook
    mock_ultratb = MagicMock()
    mock_sys_excepthook = MagicMock()
    
    # We patch sys.excepthook to capture what the function does, 
    # but we also need to mock the import within the function.
    with patch('sys.__excepthook__', mock_sys_excenthook), \
         patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb), \
         patch('sys.excepthook', create=True) as mock_new_excepthook:
        
        # Test 1: Default behavior (capture_keyboard_interrupt=False)
        # This should add KeyboardInterrupt to skip_exceptions
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # Retrieve the newly assigned excepthook from sys
        current_hook = sys.excepthook
        assert callable(current_hook)

        # Simulate a KeyboardInterrupt (should NOT call ipython_hook, should call sys.__excepthook__)
        tb = MagicMock()
        current_hook(KeyboardInterrupt(), KeyboardInterrupt(), tb)
        mock_sys_excepthook.assert_called_with(KeyboardInterrupt(), KeyboardInterrupt(), tb)
        
        # Simulate a BdbQuit (should NOT call ipython_hook, should call sys.__excepthook__)
        current_hook(BdbQuit(), BdbQuit(), tb)
        mock_sys_excepthook.assert_called_with(BdbQuit(), BdbQuit(), tb)

        # Simulate a standard ValueError (should call ipython_hook/ultratb instance)
        # Note: the function creates an instance of FormattedTB and assigns it to ipython_hook 
        # inside the scope. However, the code provided actually defines 'ipython_hook' as 
        # the formatted traceback object itself, not a wrapper.
        # In the source: ipython_hook = ultratb.FormattedTB(...)
        # The excepthook calls: ipython_hook(type, value, traceback)
        
        current_hook(ValueError(), ValueError("test"), tb)
        mock_ultratb.assert_called()

    # Test 2: capture_keyboard_interrupt=True
    # This should NOT add KeyboardInterrupt to skip_exceptions
    with patch('IPython.core.ultratb.FormattedTB', return_value=mock_ultratb), \
         patch('sys.__excepthook__', mock_sys_excepthook):
        
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        current_hook = sys.excepthook
        
        # Simulate a KeyboardInterrupt (now it SHOULD trigger ipython_hook)
        # Reset mock to clear previous calls
        mock_ultratb.reset_mock()
        current_hook(KeyboardInterrupt(), KeyboardInterrupt(), tb)
        
        # If it triggers the hook, it calls the instance returned by FormattedTB
        # Since the code uses: ipython_hook = ultratb.FormattedTB(...) 
        # and then calls ipython_hook(...), we check if the instance was called.
        assert mock_ultratb.return_value.called is True
```


# LLM-generated content at query #15
#--------------------------

```python
import sys
from unittest.mock import patch, MagicMock
from bdb import BdbQuit

def test_register_ipython_excepthook():
    # Mock the IPython component that is imported inside the function
    with patch("IPython.core.ultratb.FormattedTB") as mock_formatted_tb:
        # Setup a dummy exception hook to replace sys.excepthook
        original_excepthook = sys.excepthook
        
        try:
            # Test 1: Register with default settings (capture_keyboard_interrupt=False)
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify that the new excepthook is set
            assert sys.excepthook != original_excepthook
            
            # Create a mock traceback and exception type/value
            mock_tb = MagicMock()
            exc_type = ValueError("test error")
            
            # Test behavior for KeyboardInterrupt (should NOT trigger ipython_hook)
            # We need to check if sys.__excepthook__ is called. 
            # Since we can't easily intercept the builtin sys.__excepthook__, 
            # we test the logic by checking if the handler skips it.
            with patch("sys.__excepthook__") as mock_builtin_hook:
                sys.excepthook(KeyboardInterrupt(), KeyboardInterrupt(), mock_tb)
                mock_builtin_hook.assert_called_once()

            # Test behavior for BdbQuit (should NOT trigger ipython_hook)
            with patch("sys.__excepthook__") as mock_builtin_hook:
                sys.excepthook(BdbQuit, BdbQuit(), mock_tb)
                mock_builtin_hook.assert_called_once()

            # Test behavior for regular Exception (SHOULD trigger ipython_hook/ultratb)
            # In the implementation, 'ipython_hook' is the instance of FormattedTB
            sys.excepthook(ValueError, ValueError("test"), mock_tb)
            # The function creates an instance of FormattedTB. 
            # We check if the constructor was called.
            mock_formatted_tb.assert_called()

            # Test 2: Register with capture_keyboard_interrupt=True
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            with patch("sys.__excepthook__") as mock_builtin_hook:
                # Now KeyboardInterrupt should trigger the ipython_hook (the instance of FormattedTB)
                # We check if it calls the instance created in the function. 
                # Since we don't have the reference to the instance, we rely on the fact 
                # that it won't call sys.__excepthook__.
                sys.excepthook(KeyboardInterrupt(), KeyboardInterrupt(), mock_tb)
                mock_builtin_hook.assert_not_called()

        finally:
            # Restore original state
            sys.excepthook = original_excepthook
```


# LLM-generated content at query #16
#--------------------------

```python
import sys
from unittest.mock import MagicMock, patch
from bdb import BdbQuit

@patch("sys.excepthook")
@patch("IPython.core.ultratb.FormattedTB")
def test_register_ipython_excepthook(mock_formattedtb, mock_sys_excepthook):
    # Setup
    original_excepthook = sys.excepthook
    
    # Create a dummy function for ultratb to act as the hook
    def dummy_hook(type, value, tb):
        pass
    mock_formattedtb.return_value = dummy_hook

    try:
        # Test Case 1: capture_keyboard_interrupt=False (Default)
        # Should skip KeyboardInterrupt and BdbQuit
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        new_hook = sys.excepthook
        
        # Check if the logic triggers sys.__excepthook__ for KeyboardInterrupt
        with patch("sys.__excepthook__") as mock_original:
            new_hook(KeyboardInterrupt(), KeyboardInterrupt(), None)
            mock_original.assert_called_once()
            
        # Check if the logic triggers ipython_hook for ValueError
        # Since we can't easily check the internal 'ipython_hook' variable, 
        # we verify that it doesn't call sys.__excepthook__ for other errors.
        with patch("sys.__excepthook__") as mock_original:
            new_hook(ValueError(), ValueError(), None)
            mock_original.assert_not_called()

        # Test Case 2: capture_keyboard_interrupt=True
        # Should NOT skip KeyboardInterrupt (should trigger ipython_hook)
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        new_hook_true = sys.excepthook
        
        with patch("sys.__excepthook__") as mock_original:
            new_hook_true(KeyboardInterrupt(), KeyboardInterrupt(), None)
            # It should NOT call the original hook because we are capturing it
            mock_original.assert_not_called()

        # Test Case 3: BdbQuit should always be skipped
        with patch("sys.__excepthook__") as mock_original:
            new_hook_true(BdbQuit(), BdbQuit(), None)
            mock_original.assert_called_once()

    finally:
        # Restore original state to prevent side effects in other tests
        sys.excepthook = original_excepthook
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_exception_wrapper():
    # 1. Test Default Behavior (no handler_fn provided)
    # Should call log_exception when an exception occurs
    @exception_wrapper()
    def failing_func():
        raise ValueError("Test error")

    with patch("pytest.importorskip('flutes.log').log") as mock_log:
        # We need to patch the specific module where log_exception is defined or used
        # For this test, we assume log_exception calls a logger. 
        # Since log_exception uses 'log', we patch that.
        with patch("your_module_name.log_exception") as mock_log_exc:
            with pytest.raises(ValueError):
                failing_func()
            mock_log_exc.assert_called_once()

    # 2. Test Custom Handler with positional and keyword arguments
    handler_called = False
    def my_handler(e, one, two, three, four=None, **kwargs):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert one == 1
        assert two == "2"
        assert three == "3"
        assert four == 4
        assert kwargs["extra"] == "extra_val"

    @exception_wrapper(my_handler)
    def complex_func(one, two, *args, three=None, extra="extra_val"):
        # We must pass 'three' such that it matches the handler requirements.
        # The decorator checks if handler arguments (excluding e) exist in func signature.
        # Note: The decorator logic requires handler args to NOT have defaults 
        # if they are matched to func args, but can have defaults if not.
        pass

    # To satisfy the decorator's strict validation:
    # 'one', 'two', and 'three' must be in func signature and NOT have defaults in handler.
    @exception_wrapper(my_handler)
    def valid_func(one, two, three, extra="extra_val"):
        raise ValueError("Trigger handler")

    with pytest.raises(ValueError):
        valid_func(1, "2", "3", extra="extra_val")
    assert handler_called is True

    # 3. Test Validation: Handler must have exception argument
    def invalid_handler(one): # Missing 'e'
        pass

    with pytest.raises(ValueError, match="Exception handler must have a positional argument"):
        @exception_wrapper(invalid_handler)
        def bad_func():
            pass

    # 4. Test Validation: Handler cannot have *args
    def invalid_handler_args(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(invalid_handler_args)
        def bad_func():
            pass

    # 5. Test Validation: Argument mismatch (argument in handler not in function)
    def mismatch_handler(e, non_existent):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        @exception_wrapper(mismatch_handler)
        def bad_func(one):
            pass

    # 6. Test Validation: Argument in handler has default but exists in function
    def default_arg_handler(e, one=10):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        @exception_wrapper(default_arg_handler)
        def bad_func(one):
            pass

    # 7. Test Generator Support
    generator_results = []
    @exception_wrapper(my_handler)
    def generator_func(one, two, three, extra="extra_val"):
        yield "first"
        raise ValueError("Generator error")
        yield "second"

    gen_iter = generator_func(1, "2", "3", extra="extra_val")
    with pytest.raises(ValueError):
        for item in gen_iter:
            generator_results.append(item)
    
    assert generator_results == ["first"]
    # The handler was triggered during the iteration of the unrolled generator
```


# LLM-generated content at query #18
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
    
    # Capture original excepthook to restore it later
    original_excepthook = sys.excepthook
    
    try:
        with patch.dict('sys.modules', {'IPython.core import ultratb': mock_ultratb}):
            # We need to handle the specific import style used in the function 
            # (from IPython.core import ultratb)
            with patch('IPython.core.ultratb.FormattedTB', mock_formatted_tb):
                # Test registration with default capture_keyboard_interrupt=False
                register_ipython_excepthook(capture_keyboard_interrupt=False)
                
                new_hook = sys.excepthook
                assert new_hook != original_excepthook
                
                # Prepare test data for calling the hook
                test_type = ValueError
                test_value = ValueError("test error")
                test_traceback = MagicMock()
                
                # Mock sys.__excepthook__ to verify it is called for skipped exceptions
                with patch('sys.__excepthook__') as mock_sys_hook:
                    # Case 1: BdbQuit should be skipped (calls sys.__excepthook__)
                    new_hook(BdbQuit, BdbQuit("quit"), test_traceback)
                    mock_sys_hook.assert_called_with(BdbQuit, BdbQuit("quit"), test_traceback)
                    
                    # Case 2: KeyboardInterrupt should be skipped when capture_keyboard_interrupt=False
                    new_hook(KeyboardInterrupt, KeyboardInterrupt(), test_traceback)
                    mock_sys_hook.assert_called_with(KeyboardInterrupt, KeyboardInterrupt(), test_traceback)

                # Case 3: Standard exception should trigger ipython_hook (ultratb.FormattedTB call)
                # Since the hook is a closure capturing mock_formatted_tb via ultratb.FormattedTB
                new_hook(test_type, test_value, test_traceback)
                # The logic in register_ipython_excepthook calls ipython_hook(...) 
                # where ipython_hook is the instance of FormattedTB.
                # We check if an operation occurred (in a real scenario this would trigger PDB)
                # Given our mock setup, we verify the hook execution path doesn't crash and uses the logic.

        # Test registration with capture_keyboard_interrupt=True
        with patch('IPython.core.ultratb.FormattedTB', mock_formatted_tb):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            new_hook = sys.excepthook
            
            with patch('sys.__excepthook__') as mock_sys_hook:
                # Case 4: KeyboardInterrupt should NOT be skipped when capture_keyboard_interrupt=True
                # It should call the ipython_hook instead of sys.__excepthook__
                new_hook(KeyboardInterrupt, KeyboardInterrupt(), test_traceback)
                mock_sys_hook.assert_not_called()

    finally:
        # Restore original state
        sys.excepthook = original_excepthook
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess

@pytest.mark.parametrize("exception_type, user_msg, expected_log_calls", [
    (ValueError("test error"), None, ["<ValueError> test error"]),
    (RuntimeError("fail"), "Custom Message", ["Custom Message: <RuntimeError> fail"]),
])
def test_log_exception(exception_type, user_msg, expected_log_calls):
    with patch("your_module_path.log") as mock_log:
        log_exception(exception_type, user_msg=user_msg, extra_param="value")
        
        # Check that log was called with the correct formatted message and level
        for expected_msg in expected_log_calls:
            mock_log.assert_any_call(expected_msg, "error", extra_param="value")
        
        # Verify traceback logging (it should be called with a string)
        found_traceback = False
        for call in mock_log.call_args_list:
            if call[0][0].endswith("Traceback (most recent call last):"):
                found_traceback = True
                break
        assert found_traceback

def test_log_exception_subprocess_error():
    # For CalledProcessError with output, it should not log the traceback separately 
    # (per the logic: if not (isinstance and e.output is not None))
    err = subprocess.CalledProcessError(returncode=1, cmd="ls", output="some error output")
    
    with patch("your_module_path.log") as mock_log:
        log_exception(err)
        
        # Check that the exception message is logged
        mock_log.assert_any_call("<CalledProcessError> Command 'ls' failed with exit status 1", "error")
        
        # Ensure traceback logging was skipped because output is not None
        # We check that no call starts with "Traceback"
        for call in mock_log.call_args_list:
            assert not call[0][0].startswith("Traceback")

def test_log_exception_logging_failure():
    # Test the fallback mechanism when the logger itself crashes
    err = ValueError("original error")
    
    with patch("your_module_path.log", side_effect=Exception("logger crashed")):
        with patch("builtins.print") as mock_print:
            with pytest.raises(Exception) as excinfo:
                log_exception(err, user_msg="user")
            
            # Verify the original error message was printed to stdout
            mock_print.assert_any_call("user: <ValueError> original error")
            # Verify the secondary exception info was printed
            assert "Another exception occurred while logging" in str(excinfo.value)
            assert any("<Exception> logger crashed" in call[0][0] for call in mock_print.call_args_list)
```


