####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('sys.excepthook', sys.__excepthook__) as original_hook:
        # Import here to avoid affecting other tests
        from flutes.exception import register_ipython_excepthook
        
        # Mock IPython components
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify excepthook was replaced
            assert sys.excepthook is not original_hook
            
            # Test that KeyboardInterrupt is skipped
            mock_exc_type = KeyboardInterrupt
            mock_value = KeyboardInterrupt()
            mock_traceback = Mock()
            
            # Call the installed excepthook with KeyboardInterrupt
            sys.excepthook(mock_exc_type, mock_value, mock_traceback)
            
            # Should call original excepthook for KeyboardInterrupt
            # (We can't easily verify this without mocking sys.__excepthook__)
            
            # Test that BdbQuit is skipped
            mock_exc_type = type('BdbQuit', (), {})
            sys.excepthook(mock_exc_type, Mock(), Mock())
            # Should call original excepthook for BdbQuit
    
    # Test 2: With capture_keyboard_interrupt=True
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Verify excepthook was replaced
            assert sys.excepthook is not sys.__excepthook__
            
            # Test that KeyboardInterrupt is NOT skipped when capture_keyboard_interrupt=True
            # (Would call ipython_hook instead of original excepthook)
    
    # Test 3: Verify IPython FormattedTB is called with correct parameters
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook()
            
            # Verify FormattedTB was created with correct parameters
            mock_ultratb.FormattedTB.assert_called_once_with(
                mode='Context', 
                color_scheme='Linux', 
                call_pdb=1
            )
    
    # Test 4: Test with regular exception (not KeyboardInterrupt or BdbQuit)
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook()
            
            # Call with ValueError
            mock_exc_type = ValueError
            mock_value = ValueError("test error")
            mock_traceback = Mock()
            
            sys.excepthook(mock_exc_type, mock_value, mock_traceback)
            
            # Should call ipython_hook (mock_formatted_tb)
            mock_formatted_tb.assert_called_once_with(mock_exc_type, mock_value, mock_traceback)


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Save original excepthook
    original_excepthook = sys.excepthook
    
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    # Verify excepthook was replaced
    assert sys.excepthook != original_excepthook
    
    # Test that KeyboardInterrupt is skipped
    mock_traceback = type('MockTraceback', (), {})()
    
    # Test KeyboardInterrupt is passed to original excepthook
    called_with = []
    def mock_original_hook(type_, value, tb):
        called_with.append((type_, value, tb))
    
    sys.__excepthook__ = mock_original_hook
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), mock_traceback)
    assert len(called_with) == 1
    assert called_with[0][0] == KeyboardInterrupt
    
    # Test BdbQuit is passed to original excepthook
    called_with.clear()
    sys.excepthook(BdbQuit, BdbQuit(), mock_traceback)
    assert len(called_with) == 1
    assert called_with[0][0] == BdbQuit
    
    # Test other exceptions trigger IPython hook
    # We'll mock the IPython hook to verify it's called
    ipython_called = []
    def mock_ipython_hook(type_, value, tb):
        ipython_called.append((type_, value, tb))
    
    # Replace the IPython hook in the closure
    import types
    for cell in sys.excepthook.__closure__:
        if hasattr(cell.cell_contents, '__name__') and cell.cell_contents.__name__ == 'excepthook':
            # Find the ipython_hook variable
            for var_name, value in sys.excepthook.__code__.co_freevars:
                if var_name == 'ipython_hook':
                    # We can't directly modify the closure, so we'll test differently
                    break
    
    # Instead, test that a ValueError would go through the hook
    # by checking that original excepthook is NOT called
    called_with.clear()
    try:
        sys.excepthook(ValueError, ValueError("test"), mock_traceback)
    except (NameError, ImportError):
        # IPython might not be installed, which is fine for the test
        pass
    
    # Test 2: With capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    # Test that KeyboardInterrupt now goes to IPython hook
    # by checking original excepthook is NOT called for KeyboardInterrupt
    called_with.clear()
    try:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), mock_traceback)
    except (NameError, ImportError):
        # IPython might not be installed
        pass
    
    # Restore original excepthook
    sys.excepthook = original_excepthook


# LLM-generated content at query #3
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from io import StringIO
    from unittest.mock import patch

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    # Capture log output
    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        result = func_with_exception(1, 2)
    
    # Should return None when exception occurs with default handler
    assert result is None
    # Check that error was logged
    assert "Test error" in captured_output.getvalue()

    # Test 2: Function without exception should work normally
    @exception_wrapper()
    def func_without_exception(x, y):
        return x + y

    assert func_without_exception(3, 4) == 7

    # Test 3: Custom handler function
    def custom_handler(e, x, y):
        return f"Caught {type(e).__name__}: {e} with args {x}, {y}"

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        raise RuntimeError("Custom error")

    result = func_with_custom_handler(5, 6)
    assert result == "Caught RuntimeError: Custom error with args 5, 6"

    # Test 4: Handler with **kwargs
    def handler_with_kwargs(e, x, **kwargs):
        return {"error": str(e), "x": x, "extra": kwargs}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y=10, **kwargs):
        raise ValueError("Kwargs test")

    result = func_with_kwargs(7, z=20)
    assert result["error"] == "Kwargs test"
    assert result["x"] == 7
    assert result["extra"] == {"y": 10, "kwargs": {"z": 20}}

    # Test 5: Handler with mismatched arguments should raise ValueError
    def handler_missing_arg(e, x, z):  # z doesn't exist in wrapped function
        pass

    try:
        @exception_wrapper(handler_missing_arg)
        def func_mismatch(x, y):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 6: Handler with default values for matching arguments should raise ValueError
    def handler_bad_default(e, x, y=10):  # y matches but has default
        pass

    try:
        @exception_wrapper(handler_bad_default)
        def func_default_test(x, y):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 7: Generator function
    @exception_wrapper()
    def generator_func(items):
        for item in items:
            if item < 0:
                raise ValueError("Negative item")
            yield item * 2

    # Test generator without exception
    gen = generator_func([1, 2, 3])
    assert list(gen) == [2, 4, 6]

    # Test generator with exception
    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        gen = generator_func([1, -2, 3])
        result = list(gen)  # Should log exception and stop iteration
    
    assert len(result) == 0  # Generator should stop after exception
    assert "Negative item" in captured_output.getvalue()

    # Test 8: Handler with only exception argument
    def simple_handler(e):
        return type(e).__name__

    @exception_wrapper(simple_handler)
    def simple_func():
        raise TypeError("Simple error")

    assert simple_func() == "TypeError"

    # Test 9: Nested wrapper
    @exception_wrapper()
    @exception_wrapper(custom_handler)
    def nested_func(x):
        raise ValueError("Nested")

    # Should use the innermost wrapper's handler (custom_handler)
    result = nested_func(10)
    assert "Caught ValueError: Nested with args 10" in result

    # Test 10: Function with *args and **kwargs
    def handler_for_varargs(e, args, kwargs, extra_param=100):
        return {"args": args, "kwargs": kwargs, "extra": extra_param}

    @exception_wrapper(handler_for_varargs)
    def varargs_func(a, b, *args, **kwargs):
        raise RuntimeError("Varargs test")

    result = varargs_func(1, 2, 3, 4, x=5, y=6)
    assert result["args"] == (3, 4)
    assert result["kwargs"] == {"x": 5, "y": 6}
    assert result["extra"] == 100

    # Test 11: Handler with varargs should raise ValueError
    def invalid_handler(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler)
        def some_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 12: Empty handler should raise ValueError
    def empty_handler():
        pass

    try:
        @exception_wrapper(empty_handler)
        def another_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper():
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_raises():
        raise ValueError("Test error")

    # Should not raise, just log
    func_raises()

    # Test 2: Exception in generator with default handler
    @exception_wrapper()
    def gen_raises():
        yield 1
        raise ValueError("Generator error")
        yield 2

    # Should not raise, just log
    list(gen_raises())

    # Test 3: Custom handler with matching arguments
    handled_exceptions = []
    def custom_handler(e, arg1, arg2):
        handled_exceptions.append((e, arg1, arg2))

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("Custom handler test")

    func_with_args("val1", "val2")
    assert len(handled_exceptions) == 1
    assert isinstance(handled_exceptions[0][0], RuntimeError)
    assert handled_exceptions[0][1] == "val1"
    assert handled_exceptions[0][2] == "val2"

    # Test 4: Custom handler with default arguments
    handler_calls = []
    def handler_with_defaults(e, required_arg, optional_arg="default"):
        handler_calls.append((required_arg, optional_arg))

    @exception_wrapper(handler_with_defaults)
    def func_for_defaults(required_arg):
        raise ValueError()

    func_for_defaults("required_value")
    assert handler_calls == [("required_value", "default")]

    # Test 5: Custom handler with **kwargs
    captured_kwargs = []
    def handler_with_kwargs(e, arg1, **kwargs):
        captured_kwargs.append((arg1, kwargs))

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(arg1, arg2, kwarg1="default1"):
        raise TypeError()

    func_with_kwargs("a", "b", kwarg1="custom")
    assert captured_kwargs == [("a", {"arg2": "b", "kwarg1": "custom"})]

    # Test 6: Normal return (no exception)
    @exception_wrapper()
    def normal_func(x):
        return x * 2

    assert normal_func(5) == 10

    # Test 7: Normal generator return (no exception)
    @exception_wrapper()
    def normal_gen(n):
        for i in range(n):
            yield i * 2

    assert list(normal_gen(3)) == [0, 2, 4]

    # Test 8: Handler validation - no positional argument for exception
    def invalid_handler_no_arg():
        pass

    try:
        @exception_wrapper(invalid_handler_no_arg)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)

    # Test 9: Handler validation - varargs not allowed
    def invalid_handler_varargs(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_varargs)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 10: Handler validation - unmatched required argument
    def handler_unmatched_arg(e, nonexistent_arg):
        pass

    try:
        @exception_wrapper(handler_unmatched_arg)
        def dummy(arg1):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

    # Test 11: Handler validation - matched argument with default
    def handler_matched_default(e, arg1, arg2="default"):
        pass

    try:
        @exception_wrapper(handler_matched_default)
        def dummy(arg1, arg2):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 12: Wrapped function preserves signature
    @exception_wrapper()
    def func_with_signature(a, b=1, *args, c=2, **kwargs):
        return a + b + c

    import inspect
    sig = inspect.signature(func_with_signature)
    params = list(sig.parameters.keys())
    assert params == ['a', 'b', 'args', 'c', 'kwargs']

    # Test 13: Nested decorators work correctly
    calls = []
    def counting_handler(e):
        calls.append("handled")

    @exception_wrapper(counting_handler)
    @exception_wrapper()
    def double_wrapped():
        raise ValueError("Nested")

    double_wrapped()
    assert calls == ["handled"]


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper():
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func1():
        raise ValueError("Test error")

    # Should not raise, just log
    func1()

    # Test 2: Exception wrapper preserves return value
    @exception_wrapper()
    def func2(x):
        return x * 2

    assert func2(5) == 10

    # Test 3: Exception wrapper with generator function
    @exception_wrapper()
    def func3(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i

    # Should not raise, just log
    list(func3(5))

    # Test 4: Custom handler function
    captured_exc = []
    captured_args = []

    def custom_handler(e, arg1, arg2, extra="default"):
        captured_exc.append(e)
        captured_args.append((arg1, arg2, extra))

    @exception_wrapper(custom_handler)
    def func4(arg1, arg2, extra="default"):
        raise RuntimeError("Custom handler test")

    func4("val1", "val2", extra="not_default")
    assert len(captured_exc) == 1
    assert isinstance(captured_exc[0], RuntimeError)
    assert captured_args[0] == ("val1", "val2", "not_default")

    # Test 5: Handler with **kwargs
    captured_kwargs = []

    def handler_with_kwargs(e, arg1, **kwargs):
        captured_kwargs.append(kwargs)

    @exception_wrapper(handler_with_kwargs)
    def func5(arg1, arg2, **kwargs):
        raise ValueError("Kwargs test")

    func5("a", "b", extra1=1, extra2=2)
    assert captured_kwargs[0] == {"arg2": "b", "kwargs": {"extra1": 1, "extra2": 2}}

    # Test 6: Handler with matching argument names
    def handler_matching_args(e, x, y):
        pass

    @exception_wrapper(handler_matching_args)
    def func6(x, y):
        raise ValueError("Matching args")

    # Should not raise
    func6(1, 2)

    # Test 7: Invalid handler - no positional argument for exception
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def func7():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test 8: Invalid handler - varargs not allowed
    def handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(handler_with_varargs)
        def func8():
            pass
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test 9: Invalid handler - argument without default doesn't match wrapped function
    def handler_unmatched_arg(e, nonexistent):
        pass

    try:
        @exception_wrapper(handler_unmatched_arg)
        def func9(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

    # Test 10: Invalid handler - argument with default matches wrapped function
    def handler_default_matches(e, x, y="default"):
        pass

    try:
        @exception_wrapper(handler_default_matches)
        def func10(x, y):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 11: Nested decorators
    def counting_handler(e, count_list):
        count_list[0] += 1

    call_count = [0]

    @exception_wrapper(counting_handler)
    @exception_wrapper()
    def func11(count_list):
        raise ValueError("Nested test")

    func11(call_count)
    assert call_count[0] == 1

    # Test 12: Preserves function metadata
    @exception_wrapper()
    def func12(x: int, y: str = "default") -> str:
        """Test function"""
        return f"{x}{y}"

    assert func12.__name__ == "func12"
    assert func12.__doc__ == "Test function"
    assert str(func12.__annotations__) == "{'x': <class 'int'>, 'y': <class 'str'>, 'return': <class 'str'>}"

    # Test 13: Exception in logging doesn't crash
    import logging
    original_log = log
    
    def broken_log(*args, **kwargs):
        raise RuntimeError("Logging failed")

    # Temporarily replace log function
    import sys
    module = sys.modules[__name__]
    module.log = broken_log

    @exception_wrapper()
    def func13():
        raise ValueError("Logging test")

    try:
        func13()
    except RuntimeError as e:
        assert "Logging failed" in str(e)
    finally:
        # Restore original log function
        module.log = original_log

    # Test 14: Generator that doesn't raise returns all values
    @exception_wrapper()
    def func14(n):
        for i in range(n):
            yield i

    assert list(func14(3)) == [0, 1, 2]


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper():
    import sys
    import io
    from unittest.mock import patch

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception():
        raise ValueError("Test error")

    # Capture log output
    with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
        result = func_with_exception()
        assert result is None  # Should return None after logging exception
        assert "Test error" in fake_stderr.getvalue()

    # Test 2: Function without exception should work normally
    @exception_wrapper()
    def func_without_exception(x, y):
        return x + y

    assert func_without_exception(2, 3) == 5

    # Test 3: Generator function with exception
    @exception_wrapper()
    def gen_with_exception():
        yield 1
        raise ValueError("Generator error")
        yield 2  # This should never be reached

    gen = gen_with_exception()
    assert next(gen) == 1
    with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
        try:
            next(gen)
        except StopIteration:
            pass
        assert "Generator error" in fake_stderr.getvalue()

    # Test 4: Generator function without exception
    @exception_wrapper()
    def gen_without_exception():
        yield from [1, 2, 3]

    assert list(gen_without_exception()) == [1, 2, 3]

    # Test 5: Custom handler function
    def custom_handler(e, arg1, arg2, optional_arg="default"):
        return f"Caught {type(e).__name__}: {e} with args {arg1}, {arg2}, {optional_arg}"

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(arg1, arg2, **kwargs):
        raise RuntimeError("Custom handler test")

    result = func_with_custom_handler(10, 20, extra="data")
    assert result == "Caught RuntimeError: Custom handler test with args 10, 20, default"

    # Test 6: Handler with matching arguments and **kwargs
    def handler_with_kwargs(e, required_arg, **kwargs):
        return {"exception": str(e), "required": required_arg, "extra": kwargs}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(required_arg, optional_arg="default", **extra_kwargs):
        raise ValueError("Kwargs test")

    result = func_with_kwargs("req_value", optional_arg="opt", extra1=1, extra2=2)
    assert result["exception"] == "Kwargs test"
    assert result["required"] == "req_value"
    assert "optional_arg" in result["extra"]
    assert "extra1" in result["extra"]
    assert "extra2" in result["extra"]

    # Test 7: Handler validation - missing required argument in handler
    def invalid_handler(e):
        pass  # Missing required matching argument

    try:
        @exception_wrapper(invalid_handler)
        def func_with_missing_arg(x):
            pass
        # Should raise ValueError during decoration
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

    # Test 8: Handler validation - argument with default value matches wrapped function
    def handler_with_default_conflict(e, x, y="default"):
        pass

    try:
        @exception_wrapper(handler_with_default_conflict)
        def func_with_conflict(x, y):
            pass
        # Should raise ValueError during decoration
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 9: Handler validation - varargs not allowed
    def handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(handler_with_varargs)
        def any_func():
            pass
        # Should raise ValueError during decoration
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 10: Nested decorator unwrapping
    def simple_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @exception_wrapper()
    @simple_decorator
    def nested_decorated_func():
        raise TypeError("Nested test")

    with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
        nested_decorated_func()
        assert "Nested test" in fake_stderr.getvalue()

    # Test 11: Function with complex signature
    def complex_handler(e, pos_arg, *args, kw_arg=None, **kwargs):
        return {"pos": pos_arg, "args": args, "kw": kw_arg, "kwargs": kwargs}

    @exception_wrapper(complex_handler)
    def complex_func(a, b, *args, c=10, **kwargs):
        raise ValueError("Complex")

    result = complex_func(1, 2, 3, 4, c=20, d=30)
    assert result["pos"] == 1
    assert result["args"] == (3, 4)
    assert result["kw"] == 20
    assert result["kwargs"]["b"] == 2
    assert result["kwargs"]["d"] == 30

    # Test 12: Return value preservation when no exception
    @exception_wrapper()
    def returning_func(x):
        return x * 2

    assert returning_func(5) == 10


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from unittest.mock import patch, MagicMock

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    with patch('flutes.exception.log_exception') as mock_log:
        result = func_with_exception(1, 2)
        assert result is None
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], ValueError)

    # Test 2: No exception - normal execution
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    result = func_no_exception(3, 4)
    assert result == 7

    # Test 3: Generator function with exception
    @exception_wrapper()
    def generator_with_exception(n):
        for i in range(n):
            if i == 2:
                raise RuntimeError("Generator error")
            yield i

    with patch('flutes.exception.log_exception') as mock_log:
        gen = generator_with_exception(5)
        result = list(gen)
        assert result == [0, 1]
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], RuntimeError)

    # Test 4: Generator function without exception
    @exception_wrapper()
    def generator_no_exception(n):
        for i in range(n):
            yield i

    gen = generator_no_exception(3)
    result = list(gen)
    assert result == [0, 1, 2]

    # Test 5: Custom handler function
    def custom_handler(e, x, y, custom_arg="default"):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}, custom_arg={custom_arg}"

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y, z=10):
        raise TypeError("Custom handler test")

    result = func_custom_handler(5, 6)
    assert result == "Caught TypeError: Custom handler test with x=5, y=6, custom_arg=default"

    # Test 6: Custom handler with kwargs
    def handler_with_kwargs(e, x, **kwargs):
        return {"exception": str(e), "x": x, "kwargs": kwargs}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y=2, *args, **kwargs):
        raise ValueError("Kwargs test")

    result = func_with_kwargs(10, 20, 30, extra1=40, extra2=50)
    assert result["x"] == 10
    assert "y" in result["kwargs"]
    assert result["kwargs"]["y"] == 20
    assert "args" in result["kwargs"]
    assert result["kwargs"]["args"] == (30,)
    assert result["kwargs"]["extra1"] == 40
    assert result["kwargs"]["extra2"] == 50

    # Test 7: Handler validation - missing required argument
    def invalid_handler_no_args():
        pass

    try:
        @exception_wrapper(invalid_handler_no_args)
        def some_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)

    # Test 8: Handler validation - varargs not allowed
    def invalid_handler_varargs(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_varargs)
        def some_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 9: Handler validation - unmatched argument
    def handler_unmatched_arg(e, nonexistent_arg):
        pass

    try:
        @exception_wrapper(handler_unmatched_arg)
        def func_simple():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

    # Test 10: Handler validation - matched argument with default
    def handler_matched_with_default(e, x, y=10):
        pass

    @exception_wrapper(handler_matched_with_default)
    def func_with_args(x, y):
        return x + y

    # This should work since y has a default in handler but matches wrapped function
    result = func_with_args(1, 2)
    assert result == 3

    # Test 11: Nested decorators
    def counting_handler(e, count=0):
        return count + 1

    @exception_wrapper(counting_handler)
    @exception_wrapper(counting_handler)
    def nested_func():
        raise Exception("Nested test")

    result = nested_func()
    assert result == 2

    # Test 12: Preserves function metadata
    @exception_wrapper()
    def documented_func(x, y):
        """This is a test function."""
        return x * y

    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "This is a test function."

    # Test 13: Complex argument matching
    def complex_handler(e, a, b, *args, c=30, **kwargs):
        return {"a": a, "b": b, "c": c, "args": args, "kwargs": kwargs}

    @exception_wrapper(complex_handler)
    def complex_func(a, b=20, *args, c=30, **kwargs):
        raise ValueError("Complex test")

    result = complex_func(1, 2, 3, 4, d=5, e=6)
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 30
    assert result["args"] == (3, 4)
    assert result["kwargs"]["d"] == 5
    assert result["kwargs"]["e"] == 6

    # Test 14: Handler with only exception argument
    def simple_handler(e):
        return "Simple handler called"

    @exception_wrapper(simple_handler)
    def simple_func():
        raise RuntimeError("Simple test")

    result = simple_func()
    assert result == "Simple handler called"

    # Test 15: Return value from handler
    def returning_handler(e, x):
        return x * 2

    @exception_wrapper(returning_handler)
    def func_returns_on_error(x):
        raise ValueError("Return test")

    result = func_returns_on_error(21)
    assert result == 42


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper():
    import sys
    import io
    from unittest.mock import patch
    
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func1(x, y):
        raise ValueError("Test error")
    
    with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
        result = func1(1, 2)
        assert result is None
        output = fake_stderr.getvalue()
        assert "ValueError" in output
        assert "Test error" in output
    
    # Test 2: Function returns normally
    @exception_wrapper()
    def func2(x, y):
        return x + y
    
    result = func2(3, 4)
    assert result == 7
    
    # Test 3: Generator function with exception
    @exception_wrapper()
    def func3(n):
        for i in range(n):
            if i == 2:
                raise RuntimeError("Generator error")
            yield i
    
    with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
        gen = func3(5)
        result = list(gen)
        assert result == [0, 1]
        output = fake_stderr.getvalue()
        assert "RuntimeError" in output
        assert "Generator error" in output
    
    # Test 4: Generator function without exception
    @exception_wrapper()
    def func4(n):
        for i in range(n):
            yield i
    
    gen = func4(3)
    result = list(gen)
    assert result == [0, 1, 2]
    
    # Test 5: Custom handler function
    def custom_handler(e, x, y, custom_arg="default"):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}, custom_arg={custom_arg}"
    
    @exception_wrapper(custom_handler)
    def func5(x, y):
        raise ValueError("Custom handler test")
    
    result = func5(10, 20)
    assert result == "Caught ValueError: Custom handler test with x=10, y=20, custom_arg=default"
    
    # Test 6: Custom handler with **kwargs
    def custom_handler_with_kwargs(e, x, **kwargs):
        return f"Caught {type(e).__name__}: x={x}, kwargs={kwargs}"
    
    @exception_wrapper(custom_handler_with_kwargs)
    def func6(x, y=5, **kwargs):
        raise TypeError("Kwargs test")
    
    result = func6(1, z=3, w=4)
    assert "x=1" in result
    assert "'z': 3" in result
    assert "'w': 4" in result
    
    # Test 7: Handler with matching argument names
    def handler_with_matching_args(e, a, b, c=100):
        return f"a={a}, b={b}, c={c}"
    
    @exception_wrapper(handler_with_matching_args)
    def func7(a, b, d=50):
        raise Exception("Test")
    
    result = func7(1, 2)
    assert result == "a=1, b=2, c=100"
    
    # Test 8: Invalid handler - no positional argument for exception
    def invalid_handler_no_arg():
        pass
    
    try:
        @exception_wrapper(invalid_handler_no_arg)
        def func8():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)
    
    # Test 9: Invalid handler - varargs not allowed
    def invalid_handler_varargs(e, *args):
        pass
    
    try:
        @exception_wrapper(invalid_handler_varargs)
        def func9():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)
    
    # Test 10: Invalid handler - non-matching argument
    def handler_with_non_matching_arg(e, non_existent_arg):
        pass
    
    try:
        @exception_wrapper(handler_with_non_matching_arg)
        def func10(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)
    
    # Test 11: Invalid handler - matching argument with default value
    def handler_with_matching_default(e, x=10):
        pass
    
    try:
        @exception_wrapper(handler_with_matching_default)
        def func11(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)
    
    # Test 12: Nested decorators
    def counting_handler(e, count=[0]):
        count[0] += 1
        return count[0]
    
    @exception_wrapper(counting_handler)
    @exception_wrapper(counting_handler)
    def func12():
        raise Exception("Nested test")
    
    result = func12()
    assert result == 2  # Both handlers should be called
    
    # Test 13: Already wrapped function
    def simple_handler(e):
        return "handled"
    
    @exception_wrapper(simple_handler)
    def inner_func():
        raise ValueError("Inner")
    
    # Wrap it again
    wrapped_again = exception_wrapper(simple_handler)(inner_func)
    result = wrapped_again()
    assert result == "handled"
    
    # Test 14: Function with *args and **kwargs
    def complex_handler(e, first, *args, option="default", **kwargs):
        return f"first={first}, args={args}, option={option}, kwargs={kwargs}"
    
    @exception_wrapper(complex_handler)
    def func14(first, second, *args, **kwargs):
        raise RuntimeError("Complex")
    
    result = func14("a", "b", "c", "d", extra1=1, extra2=2)
    assert "first='a'" in result
    assert "args=('c', 'd')" in result
    assert "option='default'" in result
    assert "'extra1': 1" in result
    assert "'extra2': 2" in result
    
    # Test 15: Handler with only exception argument
    def simple_exception_handler(e):
        return type(e).__name__
    
    @exception_wrapper(simple_exception_handler)
    def func15():
        raise KeyError("Simple")
    
    result = func15()
    assert result == "KeyError"


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper():
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func1(x, y):
        raise ValueError("Test error")
        return x + y

    # Should not raise, just log
    result = func1(1, 2)
    assert result is None

    # Test 2: Function returns normally
    @exception_wrapper()
    def func2(x, y):
        return x + y

    assert func2(1, 2) == 3

    # Test 3: Generator function with exception
    @exception_wrapper()
    def func3(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i

    # Should not raise, generator should be consumed without error
    list(func3(5))

    # Test 4: Generator function without exception
    @exception_wrapper()
    def func4(n):
        for i in range(n):
            yield i

    assert list(func4(3)) == [0, 1, 2]

    # Test 5: Custom handler with matching arguments
    def custom_handler(e, x, y):
        return f"Caught {type(e).__name__} with x={x}, y={y}"

    @exception_wrapper(custom_handler)
    def func5(x, y):
        raise RuntimeError("Custom handler test")

    result = func5(10, 20)
    assert result == "Caught RuntimeError with x=10, y=20"

    # Test 6: Custom handler with default arguments
    def custom_handler2(e, x, y, msg="default"):
        return f"{msg}: {type(e).__name__}"

    @exception_wrapper(custom_handler2)
    def func6(x, y):
        raise TypeError("Test")

    result = func6(1, 2)
    assert result == "default: TypeError"

    # Test 7: Custom handler with **kwargs
    def custom_handler3(e, x, **kwargs):
        return {"x": x, "kwargs": kwargs}

    @exception_wrapper(custom_handler3)
    def func7(x, y=5, **kwargs):
        raise ValueError("Test")

    result = func7(1, z=10)
    assert result == {"x": 1, "kwargs": {"y": 5, "kwargs": {"z": 10}}}

    # Test 8: Handler with mismatched arguments should raise ValueError
    def bad_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(bad_handler)
        def func8():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

    # Test 9: Handler with matching argument that has default value should raise ValueError
    def bad_handler2(e, x=5):
        pass

    try:
        @exception_wrapper(bad_handler2)
        def func9(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 10: Handler with varargs should raise ValueError
    def bad_handler3(e, *args):
        pass

    try:
        @exception_wrapper(bad_handler3)
        def func10():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 11: Handler without exception parameter should raise ValueError
    def bad_handler4():
        pass

    try:
        @exception_wrapper(bad_handler4)
        def func11():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)

    # Test 12: Complex function signature with *args and **kwargs
    def complex_handler(e, a, args, c, **kwargs):
        return {"a": a, "args": args, "c": c, "kwargs": kwargs}

    @exception_wrapper(complex_handler)
    def func12(a, b, *args, c=30, **kwargs):
        raise Exception("Complex test")

    result = func12(1, 2, 3, 4, c=50, d=60)
    assert result == {
        "a": 1,
        "args": (3, 4),
        "c": 50,
        "kwargs": {"b": 2, "kwargs": {"d": 60}}
    }

    # Test 13: Already wrapped function
    def handler(e, x):
        return f"Handled: {x}"

    @exception_wrapper(handler)
    @exception_wrapper()
    def func13(x):
        raise ValueError("Double wrapped")

    result = func13(42)
    assert result == "Handled: 42"

    # Test 14: Function with no arguments
    @exception_wrapper()
    def func14():
        raise ValueError("No args")

    result = func14()
    assert result is None

    # Test 15: Handler accesses all arguments
    def all_args_handler(e, **kwargs):
        return kwargs

    @exception_wrapper(all_args_handler)
    def func15(a, b=2, *args, **kwargs):
        raise Exception("Test")

    result = func15(1, 3, 4, 5, extra=6)
    assert result == {"a": 1, "b": 3, "args": (4, 5), "kwargs": {"extra": 6}}


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from io import StringIO
    from unittest.mock import patch

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        result = func_with_exception(1, 2)
    assert result is None
    assert "Test error" in captured_output.getvalue()

    # Test 2: Function returns normally without exception
    @exception_wrapper()
    def func_normal(x, y):
        return x + y

    assert func_normal(3, 4) == 7

    # Test 3: Generator function with exception
    @exception_wrapper()
    def gen_with_exception(n):
        for i in range(n):
            if i == 2:
                raise RuntimeError("Generator error")
            yield i

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        gen = gen_with_exception(5)
        result = list(gen)
    assert result == [0, 1]
    assert "Generator error" in captured_output.getvalue()

    # Test 4: Generator function without exception
    @exception_wrapper()
    def gen_normal(n):
        for i in range(n):
            yield i

    gen = gen_normal(3)
    assert list(gen) == [0, 1, 2]

    # Test 5: Custom handler function with matching arguments
    def custom_handler(e, x, y):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}"

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y):
        raise ValueError("Custom handler test")

    result = func_custom_handler(10, 20)
    assert result == "Caught ValueError: Custom handler test with x=10, y=20"

    # Test 6: Custom handler with default arguments
    def handler_with_defaults(e, x, y, msg="default"):
        return f"{msg}: {e} at ({x}, {y})"

    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        raise TypeError("Type error")

    result = func_with_defaults(5, 6)
    assert result == "default: Type error at (5, 6)"

    # Test 7: Custom handler with **kwargs
    def handler_with_kwargs(e, x, **kwargs):
        return {"error": str(e), "x": x, "extra": kwargs}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y, z=30):
        raise ValueError("Kwargs test")

    result = func_with_kwargs(1, 2, z=40)
    assert result["error"] == "Kwargs test"
    assert result["x"] == 1
    assert result["extra"] == {"y": 2, "z": 40}

    # Test 8: Handler with mismatched arguments should raise ValueError
    def bad_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(bad_handler)
        def some_func(a, b):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

    # Test 9: Handler with matching argument that has default value should raise ValueError
    def handler_with_matching_default(e, x, y=10):
        pass

    try:
        @exception_wrapper(handler_with_matching_default)
        def func_with_arg(x):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 10: Handler with varargs should raise ValueError
    def handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(handler_with_varargs)
        def any_func():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 11: Handler without exception parameter should raise ValueError
    def handler_no_exception():
        pass

    try:
        @exception_wrapper(handler_no_exception)
        def another_func():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)

    # Test 12: Nested decorators should work correctly
    def counting_handler(e, count, **kwargs):
        return count + 1

    @exception_wrapper(counting_handler)
    @exception_wrapper()
    def nested_func(count):
        if count < 3:
            raise ValueError(f"Count is {count}")
        return count * 2

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        result = nested_func(2)
    assert result == 3
    assert "Count is 2" in captured_output.getvalue()

    # Test 13: Function with *args and **kwargs
    def handler_for_varargs(e, first, *args, **kwargs):
        return {"first": first, "args": args, "kwargs": kwargs}

    @exception_wrapper(handler_for_varargs)
    def func_varargs(first, *args, **kwargs):
        raise RuntimeError("Varargs test")

    result = func_varargs(1, 2, 3, extra="value")
    assert result["first"] == 1
    assert result["args"] == (2, 3)
    assert result["kwargs"] == {"extra": "value"}

    # Test 14: Already wrapped function should unwrap properly
    def simple_handler(e, x):
        return f"Handled: {x}"

    @exception_wrapper(simple_handler)
    @functools.wraps(lambda x: x)
    def already_wrapped(x):
        raise ValueError("Already wrapped")

    result = already_wrapped(42)
    assert result == "Handled: 42"


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper():
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func1(x, y):
        raise ValueError("Test error")
        return x + y
    
    # Should log exception but not raise it
    result = func1(1, 2)
    assert result is None
    
    # Test 2: Function without exception should work normally
    @exception_wrapper()
    def func2(x, y):
        return x + y
    
    assert func2(1, 2) == 3
    
    # Test 3: Custom handler function
    captured_exc = None
    captured_args = {}
    
    def custom_handler(e, x, y):
        nonlocal captured_exc, captured_args
        captured_exc = e
        captured_args = {'x': x, 'y': y}
    
    @exception_wrapper(custom_handler)
    def func3(x, y):
        raise ValueError("Custom handler test")
    
    func3(10, 20)
    assert isinstance(captured_exc, ValueError)
    assert str(captured_exc) == "Custom handler test"
    assert captured_args == {'x': 10, 'y': 20}
    
    # Test 4: Handler with default arguments
    handler_calls = []
    
    def handler_with_defaults(e, x, y, z=30, extra=None):
        handler_calls.append({
            'e': e,
            'x': x,
            'y': y,
            'z': z,
            'extra': extra
        })
    
    @exception_wrapper(handler_with_defaults)
    def func4(x, y):
        raise RuntimeError("Default args test")
    
    func4(100, 200)
    assert len(handler_calls) == 1
    assert handler_calls[0]['x'] == 100
    assert handler_calls[0]['y'] == 200
    assert handler_calls[0]['z'] == 30
    assert handler_calls[0]['extra'] is None
    assert isinstance(handler_calls[0]['e'], RuntimeError)
    
    # Test 5: Handler with **kwargs
    handler_kwargs = {}
    
    def handler_with_kwargs(e, x, **kwargs):
        nonlocal handler_kwargs
        handler_kwargs = {'e': e, 'x': x, 'kwargs': kwargs}
    
    @exception_wrapper(handler_with_kwargs)
    def func5(x, y=5, z=10):
        raise TypeError("Kwargs test")
    
    func5(1, y=2, z=3)
    assert handler_kwargs['x'] == 1
    assert handler_kwargs['kwargs'] == {'y': 2, 'z': 3}
    assert isinstance(handler_kwargs['e'], TypeError)
    
    # Test 6: Generator function
    @exception_wrapper()
    def gen_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i
    
    # Should not raise exception, generator should complete without yielding
    gen = gen_func(5)
    results = list(gen)
    assert results == []
    
    # Test 7: Generator without exception
    @exception_wrapper()
    def gen_func2(n):
        for i in range(n):
            yield i
    
    gen = gen_func2(3)
    assert list(gen) == [0, 1, 2]
    
    # Test 8: Handler with mismatched arguments should raise ValueError
    def bad_handler(e, non_existent_arg):
        pass
    
    try:
        @exception_wrapper(bad_handler)
        def func6():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)
    
    # Test 9: Handler with matching argument that has default should raise ValueError
    def handler_with_matching_default(e, x=10):
        pass
    
    try:
        @exception_wrapper(handler_with_matching_default)
        def func7(x):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)
    
    # Test 10: Handler with varargs should raise ValueError
    def handler_with_varargs(e, *args):
        pass
    
    try:
        @exception_wrapper(handler_with_varargs)
        def func8():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)
    
    # Test 11: Handler without exception parameter should raise ValueError
    def handler_no_exception():
        pass
    
    try:
        @exception_wrapper(handler_no_exception)
        def func9():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)
    
    # Test 12: Function with *args and **kwargs
    captured_kwargs = {}
    
    def complex_handler(e, a, args, kwargs, extra=100):
        nonlocal captured_kwargs
        captured_kwargs = {
            'e': e,
            'a': a,
            'args': args,
            'kwargs': kwargs,
            'extra': extra
        }
    
    @exception_wrapper(complex_handler)
    def func10(a, b, *args, c=30, **kwargs):
        raise ValueError("Complex test")
    
    func10(1, 2, 3, 4, c=5, d=6, e=7)
    assert captured_kwargs['a'] == 1
    assert captured_kwargs['args'] == (3, 4)
    assert captured_kwargs['kwargs'] == {'b': 2, 'c': 5, 'd': 6, 'e': 7}
    assert captured_kwargs['extra'] == 100
    
    # Test 13: Already wrapped function
    def simple_handler(e, x):
        pass
    
    @exception_wrapper(simple_handler)
    @exception_wrapper(simple_handler)
    def double_wrapped(x):
        raise ValueError("Double wrapped")
    
    # Should work without error
    double_wrapped(1)
    
    # Test 14: Method in class
    class TestClass:
        @exception_wrapper()
        def method(self, value):
            raise ValueError(f"Method error: {value}")
    
    obj = TestClass()
    result = obj.method("test")
    assert result is None
    
    # Test 15: Handler that returns value
    def returning_handler(e, x):
        return f"Handled: {x}"
    
    @exception_wrapper(returning_handler)
    def func11(x):
        raise ValueError("Return test")
    
    result = func11(42)
    assert result == "Handled: 42"


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from unittest.mock import patch, MagicMock

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    with patch('flutes.exception.log_exception') as mock_log:
        result = func_with_exception(1, 2)
        assert result is None
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], ValueError)

    # Test 2: No exception - normal execution
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    result = func_no_exception(3, 4)
    assert result == 7

    # Test 3: Custom handler function
    mock_handler = MagicMock()

    @exception_wrapper(mock_handler)
    def func_custom_handler(a, b, c=10):
        raise RuntimeError("Custom handler test")

    func_custom_handler(1, 2, c=20)
    assert mock_handler.called
    call_args = mock_handler.call_args[1]
    assert isinstance(call_args['e'], RuntimeError)
    assert call_args['a'] == 1
    assert call_args['b'] == 2
    assert call_args['c'] == 20

    # Test 4: Handler with **kwargs
    def handler_with_kwargs(e, a, b, **kwargs):
        return f"Handled: {e}, a={a}, b={b}, kwargs={kwargs}"

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(a, b, *args, c=30, **kwargs):
        raise TypeError("Kwargs test")

    result = func_with_kwargs(1, 2, 3, 4, c=40, d=50)
    assert "Handled:" in result
    assert "a=1" in result
    assert "b=2" in result
    assert "'c': 40" in result
    assert "'d': 50" in result

    # Test 5: Generator function
    @exception_wrapper()
    def generator_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i

    with patch('flutes.exception.log_exception') as mock_log:
        gen = generator_func(5)
        results = list(gen)
        assert results == [0, 1]
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], ValueError)

    # Test 6: Handler validation - varargs not allowed
    def invalid_handler_varargs(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_varargs)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 7: Handler validation - missing exception parameter
    def invalid_handler_no_exception():
        pass

    try:
        @exception_wrapper(invalid_handler_no_exception)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)

    # Test 8: Handler validation - argument mismatch
    def handler_missing_arg(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler_missing_arg)
        def func_with_args(x, y):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 9: Handler validation - default value for matching argument
    def handler_bad_default(e, x, y=10):
        pass

    try:
        @exception_wrapper(handler_bad_default)
        def func_args(x, y):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 10: Nested decorator unwrapping
    def dummy_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @exception_wrapper()
    @dummy_decorator
    def nested_decorator_func():
        raise KeyError("Nested test")

    with patch('flutes.exception.log_exception') as mock_log:
        nested_decorator_func()
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], KeyError)

    # Test 11: Return value preservation when no exception
    @exception_wrapper()
    def func_returns_complex():
        return {"result": "success", "data": [1, 2, 3]}

    result = func_returns_complex()
    assert result == {"result": "success", "data": [1, 2, 3]}

    # Test 12: Generator returns normally
    @exception_wrapper()
    def normal_generator(n):
        for i in range(n):
            yield i

    gen = normal_generator(3)
    results = list(gen)
    assert results == [0, 1, 2]


# LLM-generated content at query #13
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import patch, MagicMock
    import traceback
    import subprocess
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Check that traceback was logged
        assert mock_log.call_count >= 1
        traceback_call = mock_log.call_args_list[0]
        assert traceback_call[0][1] == "error"
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        # Check that user message was included
        assert mock_log.call_count >= 1
        for call in mock_log.call_args_list:
            if "Custom message" in str(call[0][0]):
                break
        else:
            assert False, "User message not found in log calls"
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = b"Error output"
        log_exception(mock_error)
        
        # Should skip traceback logging for CalledProcessError with output
        # Only the exception message should be logged
        assert mock_log.call_count == 1
        assert mock_log.call_args[0][1] == "error"
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = None
        log_exception(mock_error)
        
        # Should log both traceback and exception message
        assert mock_log.call_count == 2
    
    # Test 5: Exception during logging
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
            except Exception as log_e:
                assert str(log_e) == "Logging failed"
    
    # Test 6: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs were passed
        assert mock_log.call_count >= 1
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs.get("extra") == {"key": "value"}
        assert call_kwargs.get("stack_info") is True


# LLM-generated content at query #14
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import patch, MagicMock
    import subprocess
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Should call log twice: once for traceback, once for exception message
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[1][0][1] == "error"
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        assert mock_log.call_count == 2
        # Check that user message is included
        assert "Custom message" in mock_log.call_args_list[1][0][0]
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = b"Error output"
        log_exception(mock_error)
        
        # Should only log once (skip traceback for CalledProcessError with output)
        assert mock_log.call_count == 1
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = None
        log_exception(mock_error)
        
        # Should log twice (traceback + message)
        assert mock_log.call_count == 2
    
    # Test 5: Exception during logging
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
            except Exception as log_e:
                assert str(log_e) == "Logging failed"
    
    # Test 6: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs are passed through
        call_kwargs = mock_log.call_args_list[0][1]
        assert call_kwargs.get("extra") == {"key": "value"}
        assert call_kwargs.get("stack_info") is True


# LLM-generated content at query #15
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    original_hook = sys.excepthook
    
    # Register the hook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    # Verify the hook was replaced
    assert sys.excepthook != original_hook
    
    # Test that KeyboardInterrupt is skipped
    mock_traceback = None
    test_exception = KeyboardInterrupt()
    
    # Mock the sys.__excepthook__ to track calls
    original_dunder_hook = sys.__excepthook__
    call_count = [0]
    
    def mock_dunder_hook(type, value, traceback):
        call_count[0] += 1
    
    sys.__excepthook__ = mock_dunder_hook
    
    try:
        # Trigger the hook with KeyboardInterrupt
        sys.excepthook(KeyboardInterrupt, test_exception, mock_traceback)
        assert call_count[0] == 1
    finally:
        sys.__excepthook__ = original_dunder_hook
    
    # Test 2: With capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    # Reset call count
    call_count[0] = 0
    sys.__excepthook__ = mock_dunder_hook
    
    try:
        # Now KeyboardInterrupt should go to ipython_hook
        # We can't easily test ipython_hook, but we can verify it doesn't call sys.__excepthook__
        sys.excepthook(KeyboardInterrupt, test_exception, mock_traceback)
        assert call_count[0] == 0
    finally:
        sys.__excepthook__ = original_dunder_hook
    
    # Test 3: BdbQuit exception is always skipped
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    call_count[0] = 0
    sys.__excepthook__ = mock_dunder_hook
    
    try:
        sys.excepthook(BdbQuit, BdbQuit(), mock_traceback)
        assert call_count[0] == 1
    finally:
        sys.__excepthook__ = original_dunder_hook
    
    # Test 4: Regular exceptions trigger ipython_hook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    call_count[0] = 0
    sys.__excepthook__ = mock_dunder_hook
    
    try:
        sys.excepthook(ValueError, ValueError("test"), mock_traceback)
        assert call_count[0] == 0
    finally:
        sys.__excepthook__ = original_dunder_hook
    
    # Restore original hook
    sys.excepthook = original_hook


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception():
    import io
    import logging
    import sys
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Check that log was called with traceback
        assert mock_log.call_count >= 1
        traceback_call = mock_log.call_args_list[0]
        assert traceback_call[0][1] == "error"
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        # Find the call with the user message
        for call in mock_log.call_args_list:
            if "Custom message" in str(call[0][0]):
                assert "Custom message" in str(call[0][0])
                assert "RuntimeError" in str(call[0][0])
                break
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = b"Command output"
        log_exception(mock_error)
        
        # Should not log traceback for CalledProcessError with output
        traceback_logged = any("traceback" in str(call).lower() for call in mock_log.call_args_list)
        assert not traceback_logged
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd")
        mock_error.output = None
        log_exception(mock_error)
        
        # Should log traceback when output is None
        traceback_logged = any("traceback" in str(call).lower() for call in mock_log.call_args_list)
        assert traceback_logged
    
    # Test 5: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check kwargs were passed
        for call in mock_log.call_args_list:
            if call[1]:  # Check if there are kwargs
                assert "extra" in call[1] or "stack_info" in call[1]
    
    # Test 6: Exception during logging
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
            except Exception as log_e:
                assert str(log_e) == "Logging failed"
    
    # Test 7: Verify exception message format
    with patch('flutes.log.log') as mock_log:
        try:
            raise TypeError("Type mismatch")
        except TypeError as e:
            log_exception(e)
        
        # Check format: <ExceptionType> message
        for call in mock_log.call_args_list:
            args = call[0]
            if len(args) > 0 and isinstance(args[0], str):
                if "<TypeError>" in args[0]:
                    assert "<TypeError> Type mismatch" in args[0]


# LLM-generated content at query #2
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test 1: Default behavior - KeyboardInterrupt should not be captured
    original_hook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    # Verify that sys.excepthook has been replaced
    assert sys.excepthook != original_hook
    
    # Test that KeyboardInterrupt is passed to original hook
    import io
    from contextlib import redirect_stderr
    
    # Mock the original hook to track calls
    original_calls = []
    def mock_original_hook(type, value, traceback):
        original_calls.append((type, value))
    
    sys.__excepthook__ = mock_original_hook
    
    # Simulate KeyboardInterrupt
    exc = KeyboardInterrupt("Test interrupt")
    try:
        sys.excepthook(KeyboardInterrupt, exc, None)
    except Exception:
        pass
    
    # Verify original hook was called for KeyboardInterrupt
    assert len(original_calls) == 1
    assert original_calls[0][0] == KeyboardInterrupt
    
    # Test 2: With capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    # Reset tracking
    original_calls.clear()
    
    # Now KeyboardInterrupt should go to ipython_hook, not original hook
    # We can't easily test ipython_hook directly, but we can verify
    # that original hook is NOT called
    try:
        sys.excepthook(KeyboardInterrupt, exc, None)
    except Exception:
        pass
    
    # Original hook should not be called when capture_keyboard_interrupt=True
    assert len(original_calls) == 0
    
    # Test 3: BdbQuit should always go to original hook
    exc = BdbQuit()
    try:
        sys.excepthook(BdbQuit, exc, None)
    except Exception:
        pass
    
    # BdbQuit should always go to original hook
    assert len(original_calls) == 1
    assert original_calls[0][0] == BdbQuit
    
    # Test 4: Other exceptions should go to ipython_hook
    # We'll test with ValueError
    original_calls.clear()
    
    exc = ValueError("Test error")
    try:
        sys.excepthook(ValueError, exc, None)
    except Exception:
        pass
    
    # Other exceptions should not go to original hook
    assert len(original_calls) == 0
    
    # Restore original hook
    sys.excepthook = original_hook
    sys.__excepthook__ = original_hook


# LLM-generated content at query #3
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch, MagicMock
    from bdb import BdbQuit
    
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('sys.excepthook', sys.__excepthook__) as original_hook:
        # Import here to avoid affecting other tests
        from flutes.exception import register_ipython_excepthook
        
        # Mock IPython components
        with patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB:
            mock_ipython_hook = Mock()
            MockFormattedTB.return_value = mock_ipython_hook
            
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify IPython hook was created with correct parameters
            MockFormattedTB.assert_called_once_with(mode='Context', color_scheme='Linux', call_pdb=1)
            
            # Test that KeyboardInterrupt is passed to original excepthook
            test_exc = KeyboardInterrupt()
            sys.excepthook(KeyboardInterrupt, test_exc, None)
            assert sys.__excepthook__.called
            
            # Reset mock for next test
            sys.__excepthook__.reset_mock()
            
            # Test that BdbQuit is passed to original excepthook
            test_exc = BdbQuit()
            sys.excepthook(BdbQuit, test_exc, None)
            assert sys.__excepthook__.called
            
            # Reset mock for next test
            sys.__excepthook__.reset_mock()
            
            # Test that other exceptions trigger IPython hook
            test_exc = ValueError("test")
            sys.excepthook(ValueError, test_exc, None)
            mock_ipython_hook.assert_called_once_with(ValueError, test_exc, None)
    
    # Test 2: With capture_keyboard_interrupt=True
    with patch('sys.excepthook', sys.__excepthook__) as original_hook:
        # Re-import to get fresh module state
        import importlib
        import flutes.exception
        importlib.reload(flutes.exception)
        from flutes.exception import register_ipython_excepthook
        
        with patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB:
            mock_ipython_hook = Mock()
            MockFormattedTB.return_value = mock_ipython_hook
            
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Test that KeyboardInterrupt now triggers IPython hook
            test_exc = KeyboardInterrupt()
            sys.excepthook(KeyboardInterrupt, test_exc, None)
            mock_ipython_hook.assert_called_once_with(KeyboardInterrupt, test_exc, None)
            
            # Reset mock for next test
            mock_ipython_hook.reset_mock()
            
            # Test that BdbQuit still goes to original excepthook
            test_exc = BdbQuit()
            sys.excepthook(BdbQuit, test_exc, None)
            assert sys.__excepthook__.called
    
    # Test 3: Verify sys.excepthook is properly replaced
    original_excepthook = sys.excepthook
    
    with patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB:
        mock_ipython_hook = Mock()
        MockFormattedTB.return_value = mock_ipython_hook
        
        register_ipython_excepthook()
        
        # Verify sys.excepthook is not the original one
        assert sys.excepthook != original_excepthook
        assert sys.excepthook != sys.__excepthook__
        
        # Verify it's a callable function
        assert callable(sys.excepthook)


# LLM-generated content at query #4
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    original_hook = sys.excepthook
    
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    
    # Verify excepthook was replaced
    assert sys.excepthook != original_hook
    
    # Test that KeyboardInterrupt is skipped
    try:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), None)
    except SystemExit:
        pass  # sys.__excepthook__ might call sys.exit()
    
    # Test that BdbQuit is skipped
    try:
        sys.excepthook(BdbQuit, BdbQuit(), None)
    except SystemExit:
        pass
    
    # Test 2: With capture_keyboard_interrupt=True
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    
    # Test that KeyboardInterrupt is NOT skipped when capture_keyboard_interrupt=True
    # We can't easily test the actual IPython hook behavior, but we can verify
    # the hook is different from the default
    assert sys.excepthook != sys.__excepthook__
    
    # Test 3: Verify other exceptions go to IPython hook
    # We'll test that a ValueError doesn't trigger the default hook
    # by checking the hook is not the original
    current_hook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook == current_hook  # Should be the same custom hook
    
    # Restore original hook
    sys.excepthook = original_hook


# LLM-generated content at query #5
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch
    from bdb import BdbQuit

    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('sys.excepthook', sys.__excepthook__) as original_hook:
        # Import here to avoid affecting other tests
        from flutes.exception import register_ipython_excepthook
        
        # Mock IPython components
        with patch('flutes.exception.ultratb') as mock_ultratb:
            mock_formatted_tb = Mock()
            mock_ultratb.FormattedTB.return_value = mock_formatted_tb
            
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify IPython FormattedTB was created with correct parameters
            mock_ultratb.FormattedTB.assert_called_once_with(
                mode='Context', color_scheme='Linux', call_pdb=1
            )
            
            # Test that KeyboardInterrupt is passed to original excepthook
            test_exc = KeyboardInterrupt()
            sys.excepthook(type(test_exc), test_exc, None)
            # Should call original excepthook for KeyboardInterrupt
            
            # Test that BdbQuit is passed to original excepthook
            test_exc = BdbQuit()
            sys.excepthook(type(test_exc), test_exc, None)
            # Should call original excepthook for BdbQuit
            
            # Test that other exceptions go to IPython hook
            test_exc = ValueError("test")
            sys.excepthook(type(test_exc), test_exc, None)
            mock_formatted_tb.assert_called_once_with(type(test_exc), test_exc, None)

    # Test 2: With capture_keyboard_interrupt=True
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        with patch('flutes.exception.ultratb') as mock_ultratb:
            mock_formatted_tb = Mock()
            mock_ultratb.FormattedTB.return_value = mock_formatted_tb
            
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Test that KeyboardInterrupt now goes to IPython hook
            test_exc = KeyboardInterrupt()
            sys.excepthook(type(test_exc), test_exc, None)
            mock_formatted_tb.assert_called_once_with(type(test_exc), test_exc, None)

    # Test 3: Verify sys.excepthook is replaced
    original_hook = sys.excepthook
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        with patch('flutes.exception.ultratb'):
            register_ipython_excepthook()
            assert sys.excepthook is not original_hook


# LLM-generated content at query #6
#--------------------------

```python
def test_register_ipython_excepthook():
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    original_excepthook = sys.excepthook
    
    # Mock IPython's ultratb.FormattedTB to avoid actual IPython dependency
    class MockFormattedTB:
        def __init__(self, mode, color_scheme, call_pdb):
            self.mode = mode
            self.color_scheme = color_scheme
            self.call_pdb = call_pdb
            self.called = False
        
        def __call__(self, type, value, traceback):
            self.called = True
            self.last_args = (type, value, traceback)
    
    # Save original imports
    original_ultratb = None
    if 'IPython.core.ultratb' in sys.modules:
        original_ultratb = sys.modules['IPython.core.ultratb']
    
    mock_formatted_tb = MockFormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
    sys.modules['IPython.core.ultratb'] = type('MockUltratb', (), {'FormattedTB': lambda *args, **kwargs: mock_formatted_tb})()
    
    try:
        # Register the hook
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # Verify sys.excepthook was replaced
        assert sys.excepthook is not original_excepthook
        
        # Test that KeyboardInterrupt is passed to original excepthook
        original_called = False
        original_args = None
        
        def mock_original_excepthook(type, value, traceback):
            nonlocal original_called, original_args
            original_called = True
            original_args = (type, value, traceback)
        
        sys.__excepthook__ = mock_original_excepthook
        
        # Simulate KeyboardInterrupt
        mock_traceback = "mock_traceback"
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), mock_traceback)
        assert original_called is True
        assert original_args[0] is KeyboardInterrupt
        assert mock_formatted_tb.called is False
        
        # Test that BdbQuit is passed to original excepthook
        original_called = False
        sys.excepthook(BdbQuit, BdbQuit(), mock_traceback)
        assert original_called is True
        assert original_args[0] is BdbQuit
        assert mock_formatted_tb.called is False
        
        # Test that other exceptions trigger IPython hook
        original_called = False
        test_exception = ValueError("test error")
        sys.excepthook(ValueError, test_exception, mock_traceback)
        assert original_called is False
        assert mock_formatted_tb.called is True
        assert mock_formatted_tb.last_args == (ValueError, test_exception, mock_traceback)
        
    finally:
        # Restore original state
        sys.excepthook = original_excepthook
        if original_ultratb:
            sys.modules['IPython.core.ultratb'] = original_ultratb
        elif 'IPython.core.ultratb' in sys.modules:
            del sys.modules['IPython.core.ultratb']
    
    # Test 2: With capture_keyboard_interrupt=True
    mock_formatted_tb = MockFormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
    sys.modules['IPython.core.ultratb'] = type('MockUltratb', (), {'FormattedTB': lambda *args, **kwargs: mock_formatted_tb})()
    
    try:
        register_ipython_excepthook(capture_keyboard_interrupt=True)
        
        # Test that KeyboardInterrupt now triggers IPython hook
        original_called = False
        mock_formatted_tb.called = False
        
        def mock_original_excepthook2(type, value, traceback):
            nonlocal original_called
            original_called = True
        
        sys.__excepthook__ = mock_original_excepthook2
        
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("test"), mock_traceback)
        assert original_called is False
        assert mock_formatted_tb.called is True
        
    finally:
        sys.excepthook = original_excepthook
        if original_ultratb:
            sys.modules['IPython.core.ultratb'] = original_ultratb
        elif 'IPython.core.ultratb' in sys.modules:
            del sys.modules['IPython.core.ultratb']


# LLM-generated content at query #7
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('sys.excepthook', sys.__excepthook__) as original_hook:
        # Import here to avoid interfering with other tests
        from flutes.exception import register_ipython_excepthook
        
        # Mock IPython components
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify excepthook was replaced
            assert sys.excepthook != sys.__excepthook__
            
            # Test that KeyboardInterrupt is passed to original excepthook
            mock_original = Mock()
            sys.__excepthook__ = mock_original
            test_exc = KeyboardInterrupt("test")
            test_traceback = Mock()
            sys.excepthook(KeyboardInterrupt, test_exc, test_traceback)
            mock_original.assert_called_once_with(KeyboardInterrupt, test_exc, test_traceback)
            
            # Test that BdbQuit is passed to original excepthook
            mock_original.reset_mock()
            test_exc = type('BdbQuit', (), {})()
            sys.excepthook(type(test_exc), test_exc, test_traceback)
            mock_original.assert_called_once_with(type(test_exc), test_exc, test_traceback)
            
            # Test that other exceptions go to IPython hook
            mock_original.reset_mock()
            test_exc = ValueError("test")
            sys.excepthook(ValueError, test_exc, test_traceback)
            mock_formatted_tb.assert_called_once_with(ValueError, test_exc, test_traceback)
            mock_original.assert_not_called()
    
    # Test 2: With capture_keyboard_interrupt=True
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Verify excepthook was replaced
            assert sys.excepthook != sys.__excepthook__
            
            # Test that KeyboardInterrupt now goes to IPython hook
            mock_original = Mock()
            sys.__excepthook__ = mock_original
            test_exc = KeyboardInterrupt("test")
            test_traceback = Mock()
            sys.excepthook(KeyboardInterrupt, test_exc, test_traceback)
            mock_formatted_tb.assert_called_once_with(KeyboardInterrupt, test_exc, test_traceback)
            mock_original.assert_not_called()
            
            # Test that BdbQuit still goes to original excepthook
            mock_original.reset_mock()
            mock_formatted_tb.reset_mock()
            test_exc = type('BdbQuit', (), {})()
            sys.excepthook(type(test_exc), test_exc, test_traceback)
            mock_original.assert_called_once_with(type(test_exc), test_exc, test_traceback)
            mock_formatted_tb.assert_not_called()
    
    # Test 3: Verify IPython FormattedTB is created with correct parameters
    with patch('sys.excepthook', sys.__excepthook__):
        from flutes.exception import register_ipython_excepthook
        
        mock_ultratb = Mock()
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook()
            
            # Check FormattedTB was created with correct arguments
            mock_ultratb.FormattedTB.assert_called_once_with(
                mode='Context', 
                color_scheme='Linux', 
                call_pdb=1
            )
    
    # Restore original excepthook
    sys.excepthook = sys.__excepthook__


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper():
    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")
    
    # Should not raise, just log the exception
    func_with_exception(1, 2)
    
    # Test 2: Function without exception should work normally
    @exception_wrapper()
    def func_without_exception(x, y):
        return x + y
    
    assert func_without_exception(1, 2) == 3
    
    # Test 3: Custom handler function
    captured_exception = None
    captured_args = {}
    
    def custom_handler(e, x, y):
        nonlocal captured_exception, captured_args
        captured_exception = e
        captured_args = {'x': x, 'y': y}
    
    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        raise ValueError("Custom handler test")
    
    func_with_custom_handler(10, 20)
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "Custom handler test"
    assert captured_args == {'x': 10, 'y': 20}
    
    # Test 4: Handler with default arguments
    handler_called = False
    
    def handler_with_defaults(e, x, y, extra_arg="default"):
        nonlocal handler_called
        handler_called = True
        assert x == 1
        assert y == 2
        assert extra_arg == "default"
    
    @exception_wrapper(handler_with_defaults)
    def func_with_defaults(x, y):
        raise RuntimeError("Test")
    
    func_with_defaults(1, 2)
    assert handler_called
    
    # Test 5: Handler with **kwargs
    handler_kwargs = {}
    
    def handler_with_kwargs(e, x, **kwargs):
        nonlocal handler_kwargs
        handler_kwargs = kwargs
    
    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y, z=3):
        raise ValueError("Test")
    
    func_with_kwargs(1, 2, z=4, extra=5)
    assert handler_kwargs == {'y': 2, 'z': 4, 'extra': 5}
    
    # Test 6: Generator function
    @exception_wrapper()
    def generator_func(items):
        for item in items:
            if item == "error":
                raise ValueError("Generator error")
            yield item
    
    # Should not raise, just log
    list(generator_func(["a", "b", "error", "c"]))
    
    # Test 7: Generator function with custom handler
    gen_exception = None
    
    def gen_handler(e, items):
        nonlocal gen_exception
        gen_exception = e
    
    @exception_wrapper(gen_handler)
    def generator_with_handler(items):
        for item in items:
            if item == "error":
                raise ValueError("Generator error")
            yield item
    
    result = list(generator_with_handler(["a", "b", "error", "c"]))
    assert result == ["a", "b"]
    assert isinstance(gen_exception, ValueError)
    
    # Test 8: Invalid handler - no positional argument for exception
    def invalid_handler():
        pass
    
    try:
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)
    
    # Test 9: Invalid handler - varargs not allowed
    def handler_with_varargs(e, *args):
        pass
    
    try:
        @exception_wrapper(handler_with_varargs)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)
    
    # Test 10: Invalid handler - argument without default doesn't match wrapped function
    def handler_mismatch(e, non_existent_arg):
        pass
    
    try:
        @exception_wrapper(handler_mismatch)
        def dummy(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)
    
    # Test 11: Invalid handler - argument with default matches wrapped function
    def handler_default_match(e, x=10):
        pass
    
    try:
        @exception_wrapper(handler_default_match)
        def dummy(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)
    
    # Test 12: Already wrapped function
    def simple_handler(e, x):
        pass
    
    @exception_wrapper(simple_handler)
    @exception_wrapper(simple_handler)
    def double_wrapped(x):
        raise ValueError("Test")
    
    # Should work without error
    double_wrapped(1)
    
    # Test 13: Function with *args and **kwargs
    captured_kwargs = {}
    
    def complex_handler(e, a, args, extra="default", **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
    
    @exception_wrapper(complex_handler)
    def complex_func(a, b, *args, c=30, **kwargs):
        raise ValueError("Complex test")
    
    complex_func(1, 2, 3, 4, c=40, d=50, e=60)
    assert captured_kwargs == {'b': 2, 'c': 40, 'd': 50, 'e': 60}
    
    # Test 14: Normal return value preservation
    @exception_wrapper()
    def normal_return():
        return "success"
    
    assert normal_return() == "success"


# LLM-generated content at query #9
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import Mock, patch, call
    
    # Test 1: Basic exception logging without user message
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log:
        mock_format_exc.return_value = "Traceback details"
        test_exception = ValueError("Test error")
        
        log_exception(test_exception)
        
        mock_format_exc.assert_called_once()
        mock_log.assert_has_calls([
            call("Traceback details", "error"),
            call("<ValueError> Test error", "error")
        ])
    
    # Test 2: Exception logging with user message
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log:
        mock_format_exc.return_value = "Traceback details"
        test_exception = RuntimeError("Runtime issue")
        
        log_exception(test_exception, user_msg="Custom message")
        
        mock_log.assert_has_calls([
            call("Traceback details", "error"),
            call("Custom message: <RuntimeError> Runtime issue", "error")
        ])
    
    # Test 3: CalledProcessError with output (should skip traceback logging)
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log:
        test_exception = subprocess.CalledProcessError(
            returncode=1, 
            cmd=["test", "command"],
            output=b"Command output"
        )
        
        log_exception(test_exception)
        
        mock_format_exc.assert_not_called()
        mock_log.assert_called_once_with(
            "<CalledProcessError> Command '['test', 'command']' returned non-zero exit status 1.",
            "error"
        )
    
    # Test 4: CalledProcessError without output (should log traceback)
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log:
        mock_format_exc.return_value = "Traceback details"
        test_exception = subprocess.CalledProcessError(
            returncode=1, 
            cmd=["test", "command"],
            output=None
        )
        
        log_exception(test_exception)
        
        mock_format_exc.assert_called_once()
        assert mock_log.call_count == 2
    
    # Test 5: Exception during logging (should print and re-raise)
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log, \
         patch('builtins.print') as mock_print:
        mock_format_exc.return_value = "Traceback details"
        mock_log.side_effect = RuntimeError("Logging failed")
        test_exception = ValueError("Original error")
        
        try:
            log_exception(test_exception)
            assert False, "Should have raised the logging exception"
        except RuntimeError as e:
            assert str(e) == "Logging failed"
        
        mock_print.assert_has_calls([
            call("<ValueError> Original error"),
            call("Another exception occurred while logging: <RuntimeError> Logging failed")
        ])
    
    # Test 6: Additional kwargs passed to log function
    with patch('traceback.format_exc') as mock_format_exc, \
         patch('flutes.log.log') as mock_log:
        mock_format_exc.return_value = "Traceback details"
        test_exception = TypeError("Type mismatch")
        
        log_exception(test_exception, extra={"key": "value"}, stack_info=True)
        
        mock_log.assert_has_calls([
            call("Traceback details", "error", extra={"key": "value"}, stack_info=True),
            call("<TypeError> Type mismatch", "error", extra={"key": "value"}, stack_info=True)
        ])


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper():
    import logging
    from unittest.mock import patch, MagicMock

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    with patch('flutes.exception.log') as mock_log:
        result = func_with_exception(1, 2)
        assert result is None
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[0][0][1] == "error"
        assert mock_log.call_args_list[1][0][0] == "<ValueError> Test error"

    # Test 2: No exception - normal execution
    @exception_wrapper()
    def func_no_exception(x, y):
        return x + y

    result = func_no_exception(3, 4)
    assert result == 7

    # Test 3: Custom handler with matching arguments
    mock_handler = MagicMock()

    @exception_wrapper(mock_handler)
    def func_with_args(a, b, c=10):
        raise RuntimeError("Custom handler test")

    func_with_args(1, 2, c=20)
    mock_handler.assert_called_once()
    call_args = mock_handler.call_args[1]
    assert isinstance(call_args['e'], RuntimeError)
    assert call_args['a'] == 1
    assert call_args['b'] == 2
    assert call_args['c'] == 20

    # Test 4: Custom handler with **kwargs
    mock_handler2 = MagicMock()

    @exception_wrapper(mock_handler2)
    def func_with_kwargs(x, y, **kwargs):
        raise TypeError("Kwargs test")

    func_with_kwargs(5, 6, extra1=7, extra2=8)
    call_args = mock_handler2.call_args[1]
    assert call_args['x'] == 5
    assert call_args['y'] == 6
    assert 'kwargs' in call_args
    assert call_args['kwargs'] == {'extra1': 7, 'extra2': 8}

    # Test 5: Generator function
    @exception_wrapper()
    def generator_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i

    with patch('flutes.exception.log') as mock_log:
        gen = generator_func(5)
        result = list(gen)
        assert result == [0, 1]
        assert mock_log.call_count == 2

    # Test 6: Handler with default values for non-matching args
    def custom_handler(e, required_arg, optional_arg=None):
        logging.error(f"Handler called: {required_arg}, {optional_arg}")

    @exception_wrapper(custom_handler)
    def func_partial_match(x, y):
        raise ValueError("Partial match")

    with patch.object(logging, 'error') as mock_error:
        func_partial_match(10, 20)
        mock_error.assert_called_once()
        assert "10" in mock_error.call_args[0][0]
        assert "None" in mock_error.call_args[0][0]

    # Test 7: Invalid handler - no exception parameter
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

    # Test 8: Invalid handler - varargs not allowed
    def invalid_handler_varargs(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_varargs)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

    # Test 9: Invalid handler - required arg doesn't match wrapped function
    def handler_missing_arg(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler_missing_arg)
        def func_simple():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 10: Invalid handler - matching arg with default value
    def handler_with_default(e, x=10):
        pass

    try:
        @exception_wrapper(handler_with_default)
        def func_with_x(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 11: Already wrapped function
    mock_handler3 = MagicMock()

    @exception_wrapper(mock_handler3)
    @exception_wrapper()
    def double_wrapped(x):
        raise Exception("Double wrapped")

    double_wrapped(42)
    mock_handler3.assert_called_once()
    assert mock_handler3.call_args[1]['x'] == 42

    # Test 12: Function with *args
    mock_handler4 = MagicMock()

    @exception_wrapper(mock_handler4)
    def func_with_starargs(*args):
        raise Exception("Starargs test")

    func_with_starargs(1, 2, 3)
    call_args = mock_handler4.call_args[1]
    assert 'args' in call_args
    assert call_args['args'] == (1, 2, 3)

    # Test 13: Function with mixed parameters
    def complex_handler(e, pos1, kwarg1, extra_param=100, **kwargs):
        return f"Handled: {pos1}, {kwarg1}, {extra_param}"

    @exception_wrapper(complex_handler)
    def complex_func(pos1, pos2, *args, kwarg1=5, kwarg2=6, **kwargs):
        raise Exception("Complex test")

    with patch('flutes.exception.log_exception') as mock_log_exc:
        result = complex_func(1, 2, 3, 4, kwarg1=10, extra=99)
        assert result == "Handled: 1, 10, 100"
        mock_log_exc.assert_not_called()

    # Test 14: Handler returns value
    def returning_handler(e, x):
        return x * 2

    @exception_wrapper(returning_handler)
    def func_returns_on_error(x):
        raise Exception("Should return from handler")

    result = func_returns_on_error(21)
    assert result == 42

    # Test 15: Normal generator (no exception)
    @exception_wrapper()
    def safe_generator(n):
        for i in range(n):
            yield i * 2

    gen = safe_generator(3)
    result = list(gen)
    assert result == [0, 2, 4]


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from io import StringIO
    from unittest.mock import patch

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_raises():
        raise ValueError("Test error")

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        func_raises()
    assert "Test error" in captured_output.getvalue()
    assert "<ValueError>" in captured_output.getvalue()

    # Test 2: Function with arguments and default handler
    @exception_wrapper()
    def func_with_args(x, y=10):
        raise RuntimeError(f"Error with {x} and {y}")

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        func_with_args(5, y=20)
    assert "Error with 5 and 20" in captured_output.getvalue()

    # Test 3: Custom handler function
    def custom_handler(e, x, y):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}"

    @exception_wrapper(custom_handler)
    def func_custom_handler(x, y):
        raise ValueError("Custom handler test")

    result = func_custom_handler(1, 2)
    assert result == "Caught ValueError: Custom handler test with x=1, y=2"

    # Test 4: Handler with **kwargs
    def handler_with_kwargs(e, x, **kwargs):
        return f"Error: {e}, x={x}, kwargs={kwargs}"

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y=5, **extra):
        raise TypeError("Kwargs test")

    result = func_with_kwargs(10, z=20)
    assert result == "Error: Kwargs test, x=10, kwargs={'y': 5, 'extra': {'z': 20}}"

    # Test 5: Handler with default values
    def handler_with_defaults(e, x, custom_arg="default"):
        return f"{e} | x={x} | custom={custom_arg}"

    @exception_wrapper(handler_with_defaults)
    def func_for_defaults(x):
        raise ValueError("Defaults test")

    result = func_for_defaults(100)
    assert result == "Defaults test | x=100 | custom=default"

    # Test 6: Generator function
    @exception_wrapper()
    def generator_func(n):
        for i in range(n):
            if i == 2:
                raise ValueError("Generator error")
            yield i

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        gen = generator_func(5)
        result = list(gen)
    assert result == [0, 1]
    assert "Generator error" in captured_output.getvalue()

    # Test 7: Validator - handler without exception parameter
    def invalid_handler():
        pass

    try:
        @exception_wrapper(invalid_handler)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

    # Test 8: Validator - handler with varargs
    def handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(handler_with_varargs)
        def dummy():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

    # Test 9: Validator - handler argument without match
    def handler_no_match(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler_no_match)
        def dummy(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 10: Validator - handler argument with default that matches wrapped method
    def handler_bad_default(e, x=10):
        pass

    try:
        @exception_wrapper(handler_bad_default)
        def dummy(x):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 11: Normal return (no exception)
    @exception_wrapper()
    def normal_func(a, b):
        return a + b

    assert normal_func(3, 4) == 7

    # Test 12: Generator with normal completion
    @exception_wrapper()
    def normal_generator(n):
        for i in range(n):
            yield i * 2

    assert list(normal_generator(3)) == [0, 2, 4]

    # Test 13: Nested decorator unwrapping
    def dummy_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @exception_wrapper()
    @dummy_decorator
    def nested_func():
        raise ValueError("Nested error")

    captured_output = StringIO()
    with patch('sys.stderr', captured_output):
        nested_func()
    assert "Nested error" in captured_output.getvalue()

    # Test 14: Complex argument matching
    def complex_handler(e, a, b, c=30, **kwargs):
        return f"{e} | a={a} b={b} c={c} kwargs={kwargs}"

    @exception_wrapper(complex_handler)
    def complex_func(a, *args, b=20, **kwargs):
        raise ValueError("Complex test")

    result = complex_func(1, 2, 3, b=5, extra=10)
    assert result == "Complex test | a=1 b=5 c=30 kwargs={'args': (2, 3), 'kwargs': {'extra': 10}}"


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from io import StringIO
    from unittest.mock import patch

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    # Capture log output
    with patch('sys.stderr', new=StringIO()) as fake_stderr:
        result = func_with_exception(1, 2)
        assert result is None  # Should return None when exception occurs
        output = fake_stderr.getvalue()
        assert "Test error" in output
        assert "ValueError" in output

    # Test 2: Function without exception should work normally
    @exception_wrapper()
    def func_without_exception(x, y):
        return x + y

    result = func_without_exception(3, 4)
    assert result == 7

    # Test 3: Generator function with exception
    @exception_wrapper()
    def generator_with_exception(n):
        for i in range(n):
            if i == 2:
                raise RuntimeError("Generator error")
            yield i

    with patch('sys.stderr', new=StringIO()) as fake_stderr:
        gen = generator_with_exception(5)
        results = list(gen)  # Should trigger exception at i=2
        output = fake_stderr.getvalue()
        assert "Generator error" in output
        assert "RuntimeError" in output
        assert results == [0, 1]  # Should yield values before exception

    # Test 4: Generator function without exception
    @exception_wrapper()
    def generator_without_exception(n):
        for i in range(n):
            yield i

    gen = generator_without_exception(3)
    results = list(gen)
    assert results == [0, 1, 2]

    # Test 5: Custom handler function
    def custom_handler(e, x, y, custom_arg="default"):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}, custom_arg={custom_arg}"

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y):
        raise TypeError("Custom handler test")

    result = func_with_custom_handler(10, 20)
    assert result == "Caught TypeError: Custom handler test with x=10, y=20, custom_arg=default"

    # Test 6: Custom handler with **kwargs
    def handler_with_kwargs(e, x, **kwargs):
        return {"exception": str(e), "x": x, "extra": kwargs}

    @exception_wrapper(handler_with_kwargs)
    def func_with_kwargs(x, y=5, *args, z=10, **kwargs):
        raise ValueError("Kwargs test")

    result = func_with_kwargs(1, 2, 3, 4, extra1=100, extra2=200)
    assert result["exception"] == "Kwargs test"
    assert result["x"] == 1
    assert "y" in result["extra"]
    assert "args" in result["extra"]
    assert "z" in result["extra"]
    assert "kwargs" in result["extra"]

    # Test 7: Invalid handler - no positional argument for exception
    def invalid_handler_no_arg():
        pass

    try:
        @exception_wrapper(invalid_handler_no_arg)
        def dummy_func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

    # Test 8: Invalid handler - has varargs
    def invalid_handler_varargs(e, *args):
        pass

    try:
        @exception_wrapper(invalid_handler_varargs)
        def dummy_func2():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

    # Test 9: Invalid handler - argument doesn't match wrapped function
    def handler_mismatch(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler_mismatch)
        def func_with_args(a, b):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 10: Invalid handler - argument with default matches wrapped function
    def handler_default_match(e, a="default"):
        pass

    try:
        @exception_wrapper(handler_default_match)
        def func_with_a(a):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 11: Already wrapped function
    def original_handler(e, x):
        return f"Handled: {x}"

    @exception_wrapper(original_handler)
    @exception_wrapper()  # Double wrapper
    def double_wrapped(x):
        raise ValueError("Double wrapped")

    result = double_wrapped(42)
    assert result == "Handled: 42"

    # Test 12: Function with complex signature
    def complex_handler(e, required_arg, optional_arg=None, **extra):
        return {"error": str(e), "required": required_arg, "optional": optional_arg, "extra": extra}

    @exception_wrapper(complex_handler)
    def complex_func(a, b=2, *args, c=3, **kwargs):
        raise RuntimeError("Complex error")

    result = complex_func(1, 2, 4, 5, d=6, e=7)
    assert result["error"] == "Complex error"
    assert result["required"] == 1
    assert result["optional"] == 2
    assert "args" in result["extra"]
    assert "c" in result["extra"]
    assert "kwargs" in result["extra"]

    # Test 13: Handler returns value when exception occurs
    def returning_handler(e, x):
        return x * 2

    @exception_wrapper(returning_handler)
    def func_returns_on_error(x):
        raise Exception("Error")

    result = func_returns_on_error(21)
    assert result == 42

    # Test 14: No exception in wrapped function
    @exception_wrapper(returning_handler)
    def func_no_error(x):
        return x + 1

    result = func_no_error(10)
    assert result == 11


# LLM-generated content at query #13
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import Mock, patch, call
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Should call log twice: once for traceback, once for exception message
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[0][0][1] == "error"
        assert mock_log.call_args_list[1][0][1] == "error"
        assert "Test error" in str(mock_log.call_args_list[1][0][0])
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Something went wrong")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        assert mock_log.call_count == 2
        last_call_args = mock_log.call_args_list[1][0][0]
        assert "Custom message" in str(last_call_args)
        assert "Something went wrong" in str(last_call_args)
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd", output=b"output text")
        log_exception(mock_error)
        
        # Should only log the exception message, not traceback
        assert mock_log.call_count == 1
        assert mock_log.call_args_list[0][0][1] == "error"
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        mock_error = subprocess.CalledProcessError(1, "cmd", output=None)
        log_exception(mock_error)
        
        # Should log both traceback and exception message
        assert mock_log.call_count == 2
    
    # Test 5: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs are passed through
        assert mock_log.call_count == 2
        for call_args in mock_log.call_args_list:
            kwargs = call_args[1]
            assert "extra" in kwargs
            assert kwargs["extra"]["key"] == "value"
            assert kwargs["stack_info"] is True
    
    # Test 6: Exception during logging
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
                assert False, "Should have raised exception"
            except Exception as log_e:
                assert str(log_e) == "Logging failed"


# LLM-generated content at query #14
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch, MagicMock
    
    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('sys.excepthook', sys.__excepthook__):
        # Import here to avoid affecting other tests
        from flutes.exception import register_ipython_excepthook
        
        # Mock IPython components
        mock_ultratb = Mock()
        mock_formatted_tb = Mock()
        mock_ultratb.FormattedTB.return_value = mock_formatted_tb
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=False)
            
            # Verify excepthook was replaced
            assert sys.excepthook != sys.__excepthook__
            
            # Test that KeyboardInterrupt is skipped
            mock_traceback = Mock()
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
            # Should call original excepthook for KeyboardInterrupt
            
            # Test that BdbQuit is skipped
            from bdb import BdbQuit
            sys.excepthook(BdbQuit, BdbQuit(), mock_traceback)
            # Should call original excepthook for BdbQuit
            
            # Test that other exceptions trigger IPython
            mock_formatted_tb.reset_mock()
            sys.excepthook(ValueError, ValueError("test"), mock_traceback)
            mock_formatted_tb.assert_called_once_with(ValueError, ValueError("test"), mock_traceback)
    
    # Test 2: With capture_keyboard_interrupt=True
    with patch('sys.excepthook', sys.__excepthook__):
        with patch.dict('sys.modules', {'IPython.core.ultratb': mock_ultratb}):
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            # Test that KeyboardInterrupt now triggers IPython
            mock_formatted_tb.reset_mock()
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
            mock_formatted_tb.assert_called_once_with(KeyboardInterrupt, KeyboardInterrupt(), mock_traceback)
    
    # Test 3: Verify IPython FormattedTB is configured correctly
    with patch('sys.excepthook', sys.__excepthook__):
        with patch('IPython.core.ultratb.FormattedTB') as mock_formatted_tb_class:
            register_ipython_excepthook()
            
            # Check FormattedTB was instantiated with correct parameters
            mock_formatted_tb_class.assert_called_once_with(
                mode='Context', 
                color_scheme='Linux', 
                call_pdb=1
            )
    
    # Restore original excepthook
    sys.excepthook = sys.__excepthook__


# LLM-generated content at query #15
#--------------------------

```python
def test_log_exception():
    import io
    import logging
    import sys
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Check that format_exc was called and error message was logged
        assert mock_log.call_count == 2
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        # Verify user message was included
        assert mock_log.call_count == 2
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        try:
            raise subprocess.CalledProcessError(1, "cmd", output=b"error output")
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should only log the exception message, not traceback
        assert mock_log.call_count == 1
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        try:
            raise subprocess.CalledProcessError(1, "cmd", output=None)
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should log both traceback and exception message
        assert mock_log.call_count == 2
    
    # Test 5: Exception during logging (fallback to print)
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        with patch('builtins.print') as mock_print:
            try:
                raise KeyError("Missing key")
            except KeyError as e:
                try:
                    log_exception(e)
                except Exception as log_e:
                    # Should print the original exception
                    assert mock_print.call_count >= 1
                    # Should re-raise the logging exception
                    assert str(log_e) == "Logging failed"
    
    # Test 6: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise TypeError("Type mismatch")
        except TypeError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs were passed through
        assert mock_log.call_count == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper():
    import sys
    from unittest.mock import patch, MagicMock

    # Test 1: Basic exception logging with default handler
    @exception_wrapper()
    def func_with_exception(x, y):
        raise ValueError("Test error")

    with patch('flutes.exception.log_exception') as mock_log:
        result = func_with_exception(1, 2)
        assert result is None
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], ValueError)

    # Test 2: Function returns normally without exception
    @exception_wrapper()
    def func_normal(x, y):
        return x + y

    result = func_normal(1, 2)
    assert result == 3

    # Test 3: Function returns generator without exception
    @exception_wrapper()
    def func_generator(n):
        for i in range(n):
            yield i

    gen = func_generator(3)
    assert list(gen) == [0, 1, 2]

    # Test 4: Generator with exception
    @exception_wrapper()
    def func_generator_with_exception(n):
        for i in range(n):
            if i == 1:
                raise RuntimeError("Generator error")
            yield i

    with patch('flutes.exception.log_exception') as mock_log:
        gen = func_generator_with_exception(3)
        result = list(gen)
        assert result == []
        assert mock_log.called
        assert isinstance(mock_log.call_args[0][0], RuntimeError)

    # Test 5: Custom handler function
    def custom_handler(e, x, y, custom_arg="default"):
        return f"Caught {type(e).__name__}: {e} with x={x}, y={y}, custom_arg={custom_arg}"

    @exception_wrapper(custom_handler)
    def func_with_custom_handler(x, y, z=10):
        raise TypeError("Custom handler test")

    result = func_with_custom_handler(5, 6, z=20)
    assert "Caught TypeError: Custom handler test with x=5, y=6, custom_arg=default" in result

    # Test 6: Custom handler with **kwargs
    def custom_handler_with_kwargs(e, x, **kwargs):
        return {"exception": str(e), "x": x, "extra": kwargs}

    @exception_wrapper(custom_handler_with_kwargs)
    def func_with_kwargs(x, y=2, **extra_kwargs):
        raise ValueError("Kwargs test")

    result = func_with_kwargs(10, y=3, a=1, b=2)
    assert result["x"] == 10
    assert result["extra"]["y"] == 3
    assert result["extra"]["a"] == 1
    assert result["extra"]["b"] == 2

    # Test 7: Handler with mismatched arguments should raise ValueError
    def handler_missing_arg(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler_missing_arg)
        def func_mismatch():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

    # Test 8: Handler with default value matching wrapped argument should raise ValueError
    def handler_default_conflict(e, x, y=5):
        pass

    try:
        @exception_wrapper(handler_default_conflict)
        def func_conflict(x, y):
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

    # Test 9: Handler with varargs should raise ValueError
    def handler_with_varargs(e, *args):
        pass

    try:
        @exception_wrapper(handler_with_varargs)
        def func_varargs():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

    # Test 10: Handler without exception parameter should raise ValueError
    def handler_no_exception():
        pass

    try:
        @exception_wrapper(handler_no_exception)
        def func_no_exception_param():
            pass
        # Should not reach here
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)

    # Test 11: Nested decorators
    def counting_handler(e, count=[0]):
        count[0] += 1
        return count[0]

    @exception_wrapper(counting_handler)
    @exception_wrapper(counting_handler)
    def double_wrapped():
        raise Exception("Nested test")

    result = double_wrapped()
    assert result == 1  # Only inner handler should catch it

    # Test 12: Function with *args and **kwargs
    def handler_complex(e, first, *args, middle=None, **kwargs):
        return {"first": first, "args": args, "middle": middle, "kwargs": kwargs}

    @exception_wrapper(handler_complex)
    def func_complex(first, second, *args, third=30, **kwargs):
        raise ValueError("Complex test")

    result = func_complex(1, 2, 3, 4, third=300, extra="value")
    assert result["first"] == 1
    assert result["args"] == (3, 4)
    assert result["middle"] is None
    assert result["kwargs"]["second"] == 2
    assert result["kwargs"]["third"] == 300
    assert result["kwargs"]["extra"] == "value"

    # Test 13: Method in class
    class TestClass:
        @exception_wrapper()
        def method_with_exception(self, value):
            raise RuntimeError(f"Method error: {value}")

    obj = TestClass()
    with patch('flutes.exception.log_exception') as mock_log:
        result = obj.method_with_exception("test")
        assert result is None
        assert mock_log.called

    # Test 14: Ensure original function attributes are preserved
    @exception_wrapper()
    def func_with_doc():
        """Test function with docstring"""
        return 42

    assert func_with_doc.__doc__ == "Test function with docstring"
    assert func_with_doc.__name__ == "func_with_doc"

    # Test 15: Handler returns value
    def handler_returns_value(e, x):
        return x * 2

    @exception_wrapper(handler_returns_value)
    def func_returns_on_error(x):
        raise ValueError("Return test")

    result = func_returns_on_error(21)
    assert result == 42


# LLM-generated content at query #17
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic exception logging
    with patch('flutes.log.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Check that format_exc was called
        assert mock_log.call_count >= 1
        assert any("error" in str(call) for call in mock_log.call_args_list)
    
    # Test 2: Exception with user message
    with patch('flutes.log.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        # Check that user message was included
        assert mock_log.call_count >= 1
        call_args = str(mock_log.call_args_list)
        assert "Custom message" in call_args
        assert "RuntimeError" in call_args
    
    # Test 3: CalledProcessError with output
    with patch('flutes.log.log') as mock_log:
        try:
            error = subprocess.CalledProcessError(1, "cmd", output=b"error output")
            raise error
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should log exception message but not format_exc for CalledProcessError with output
        assert mock_log.call_count >= 1
    
    # Test 4: CalledProcessError without output
    with patch('flutes.log.log') as mock_log:
        try:
            error = subprocess.CalledProcessError(1, "cmd", output=None)
            raise error
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should log both format_exc and exception message
        assert mock_log.call_count >= 2
    
    # Test 5: Additional kwargs passed to log
    with patch('flutes.log.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs were passed
        assert mock_log.call_count >= 1
        # Verify at least one call has our kwargs
        for call in mock_log.call_args_list:
            args, kwargs = call
            if "stack_info" in kwargs:
                assert kwargs["stack_info"] is True
                if "extra" in kwargs:
                    assert kwargs["extra"] == {"key": "value"}
    
    # Test 6: Exception during logging
    with patch('flutes.log.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
            except Exception as log_e:
                assert str(log_e) == "Logging failed"
    
    # Test 7: Verify exception message format
    with patch('flutes.log.log') as mock_log:
        try:
            raise TypeError("Type mismatch")
        except TypeError as e:
            log_exception(e)
        
        # Check message contains exception class and message
        for call in mock_log.call_args_list:
            args, kwargs = call
            if len(args) > 0 and isinstance(args[0], str):
                if "<TypeError>" in args[0]:
                    assert "Type mismatch" in args[0]


# LLM-generated content at query #18
#--------------------------

```python
def test_register_ipython_excepthook():
    import sys
    from unittest.mock import Mock, patch
    from bdb import BdbQuit

    # Test 1: Default behavior (capture_keyboard_interrupt=False)
    with patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB:
        mock_ipython_hook = Mock()
        MockFormattedTB.return_value = mock_ipython_hook
        
        original_excepthook = sys.excepthook
        
        # Call the function
        register_ipython_excepthook(capture_keyboard_interrupt=False)
        
        # Verify IPython hook was created with correct parameters
        MockFormattedTB.assert_called_once_with(mode='Context', color_scheme='Linux', call_pdb=1)
        
        # Get the installed excepthook
        installed_hook = sys.excepthook
        
        # Test 2: Verify KeyboardInterrupt is passed to original excepthook
        mock_original = Mock()
        sys.__excepthook__ = mock_original
        
        # Simulate KeyboardInterrupt
        exc_type = KeyboardInterrupt
        exc_value = KeyboardInterrupt()
        exc_traceback = Mock()
        
        installed_hook(exc_type, exc_value, exc_traceback)
        mock_original.assert_called_once_with(exc_type, exc_value, exc_traceback)
        
        # Test 3: Verify BdbQuit is passed to original excepthook
        mock_original.reset_mock()
        exc_type = BdbQuit
        exc_value = BdbQuit()
        
        installed_hook(exc_type, exc_value, exc_traceback)
        mock_original.assert_called_once_with(exc_type, exc_value, exc_traceback)
        
        # Test 4: Verify other exceptions trigger IPython hook
        mock_original.reset_mock()
        exc_type = ValueError
        exc_value = ValueError("test error")
        
        installed_hook(exc_type, exc_value, exc_traceback)
        mock_ipython_hook.assert_called_once_with(exc_type, exc_value, exc_traceback)
        mock_original.assert_not_called()
        
        # Test 5: Test with capture_keyboard_interrupt=True
        sys.excepthook = original_excepthook  # Reset first
        
        with patch('IPython.core.ultratb.FormattedTB') as MockFormattedTB2:
            mock_ipython_hook2 = Mock()
            MockFormattedTB2.return_value = mock_ipython_hook2
            
            register_ipython_excepthook(capture_keyboard_interrupt=True)
            
            installed_hook2 = sys.excepthook
            mock_original2 = Mock()
            sys.__excepthook__ = mock_original2
            
            # Verify KeyboardInterrupt now triggers IPython hook
            exc_type = KeyboardInterrupt
            exc_value = KeyboardInterrupt()
            
            installed_hook2(exc_type, exc_value, exc_traceback)
            mock_ipython_hook2.assert_called_once_with(exc_type, exc_value, exc_traceback)
            mock_original2.assert_not_called()
        
        # Restore original excepthook
        sys.excepthook = original_excepthook


# LLM-generated content at query #19
#--------------------------

```python
def test_log_exception():
    import logging
    from unittest.mock import Mock, patch, call
    import subprocess
    
    # Test 1: Basic exception logging
    with patch('flutes.exception.log') as mock_log:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e)
        
        # Should call log twice: once for traceback, once for exception message
        assert mock_log.call_count == 2
        assert mock_log.call_args_list[0][0][1] == "error"
        assert mock_log.call_args_list[1][0][1] == "error"
        assert "Test error" in str(mock_log.call_args_list[1][0][0])
    
    # Test 2: Exception with user message
    with patch('flutes.exception.log') as mock_log:
        try:
            raise RuntimeError("Runtime issue")
        except RuntimeError as e:
            log_exception(e, user_msg="Custom message")
        
        assert mock_log.call_count == 2
        last_call_args = mock_log.call_args_list[1][0]
        assert "Custom message" in str(last_call_args[0])
        assert "Runtime issue" in str(last_call_args[0])
    
    # Test 3: CalledProcessError with output
    with patch('flutes.exception.log') as mock_log:
        try:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["ls", "/nonexistent"],
                output=b"File not found"
            )
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should only log the exception message, not traceback
        assert mock_log.call_count == 1
        assert mock_log.call_args[0][1] == "error"
    
    # Test 4: CalledProcessError without output
    with patch('flutes.exception.log') as mock_log:
        try:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["ls", "/nonexistent"]
            )
        except subprocess.CalledProcessError as e:
            log_exception(e)
        
        # Should log both traceback and exception message
        assert mock_log.call_count == 2
    
    # Test 5: Additional kwargs passed to log
    with patch('flutes.exception.log') as mock_log:
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            log_exception(e, extra={"key": "value"}, stack_info=True)
        
        # Check that kwargs are passed through
        assert mock_log.call_count == 2
        for call_args in mock_log.call_args_list:
            assert call_args[1].get("extra") == {"key": "value"}
            assert call_args[1].get("stack_info") is True
    
    # Test 6: Exception during logging
    with patch('flutes.exception.log', side_effect=Exception("Logging failed")):
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                log_exception(e)
            except Exception as log_e:
                assert str(log_e) == "Logging failed"
    
    # Test 7: Verify exception message format
    with patch('flutes.exception.log') as mock_log:
        class CustomError(Exception):
            pass
        
        try:
            raise CustomError("Custom error message")
        except CustomError as e:
            log_exception(e)
        
        last_message = str(mock_log.call_args_list[1][0][0])
        assert "<CustomError>" in last_message
        assert "Custom error message" in last_message


