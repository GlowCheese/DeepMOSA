####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("Test error")
    user_msg = "Custom message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = TypeError("Another test error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "cmd", output="Error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("Test runtime error")
    log_exception(e, include_proc_id=False, force_console=True, timestamp=False)


# LLM-generated content at query #2
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd", output=b"error")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "my_arg": my_arg, "kw": kw}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, my_arg=10):
        raise RuntimeError("custom error")

    func_with_args(1, 2, my_arg=20, extra=30)
    assert handler_called
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["my_arg"] == 20
    assert handler_args["kw"] == {"extra": 30}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise StopIteration("generator error")

    gen = generator_that_raises()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_invalid_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda *args: None)
        def func():
            pass

    with pytest.raises(ValueError):
        def handler(e, arg):
            pass

        @exception_wrapper(handler)
        def func(other_arg):
            pass

    with pytest.raises(ValueError):
        def handler(e, arg=1):
            pass

        @exception_wrapper(handler)
        def func(arg):
            pass


# LLM-generated content at query #5
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = "some output"
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, arg2, default_arg=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, default_arg=10):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_generator():
        yield 1
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    assert next(gen) == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    list(test_gen())

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_handler_with_defaults():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_handler_with_kwargs():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False

    def custom_handler(e, arg1, arg2):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    assert test_func(1, 2) is None
    assert handler_called

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @exception_wrapper(handler)
    def test_func(a, b, d=4):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_varkw():
    handler_args = {}

    def handler(e, a, **kw):
        handler_args["a"] = a
        handler_args.update(kw)

    @exception_wrapper(handler)
    def test_func(a, b, **kwargs):
        raise ValueError("Test error")

    test_func(1, b=2, c=3)
    assert handler_args == {"a": 1, "b": 2, "kwargs": {"c": 3}}


# LLM-generated content at query #12
#--------------------------

```python
def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_without_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_always_skips_bdbquit():
    register_ipython_excepthook()
    assert BdbQuit in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_sets_excepthook():
    original_excepthook = sys.excepthook
    register_ipython_excepthook()
    assert sys.excepthook is not original_excepthook


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    skip_exceptions = [BdbQuit]
    assert not any(BdbQuit is exc_type for exc_type in skip_exceptions) is False


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_with_handler_fn():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2):
        pass

    assert True


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from IPython.core import ultratb

    original_excepthook = sys.excepthook
    register_ipython_excepthook()

    assert sys.excepthook is not original_excepthook
    assert isinstance(sys.excepthook, type(lambda: None))

    # Trigger an exception to test the hook
    try:
        raise ValueError("Test exception")
    except ValueError:
        pass

    # Restore original hook
    sys.excepthook = original_excepthook

def test_register_ipython_excepthook_with_keyboard_interrupt():
    import sys
    from IPython.core import ultratb

    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=True)

    assert sys.excepthook is not original_excepthook
    assert isinstance(sys.excepthook, type(lambda: None))

    # Restore original hook
    sys.excepthook = original_excepthook

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    import sys
    from IPython.core import ultratb

    original_excepthook = sys.excepthook
    register_ipython_excepthook(capture_keyboard_interrupt=False)

    assert sys.excepthook is not original_excepthook
    assert isinstance(sys.excepthook, type(lambda: None))

    # Trigger a KeyboardInterrupt to test the skip
    try:
        raise KeyboardInterrupt("Test keyboard interrupt")
    except KeyboardInterrupt:
        pass

    # Restore original hook
    sys.excepthook = original_excepthook


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_handler_args_mismatch():
    def custom_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError, match="does not match any argument"):
        exception_wrapper(custom_handler)(lambda: None)

def test_exception_wrapper_handler_with_defaults():
    def custom_handler(e, arg1=None):
        pass

    with pytest.raises(ValueError, match="cannot have default values"):
        exception_wrapper(custom_handler)(lambda arg1: None)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_evaluates_to_false():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert not any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True, timestamp=False)
    assert True

def test_log_exception_without_user_msg():
    e = RuntimeError("test runtime error")
    log_exception(e, force_console=True, timestamp=False)
    assert True

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, force_console=True, timestamp=False)
    assert True

def test_log_exception_with_kwargs():
    e = KeyError("test key error")
    log_exception(e, user_msg="Key not found", force_console=True, timestamp=True, include_proc_id=False)
    assert True

def test_log_exception_raises_another_exception():
    e = TypeError("test type error")
    with patch("flutes.log.log", side_effect=RuntimeError("log error")):
        try:
            log_exception(e, force_console=True, timestamp=False)
        except RuntimeError as log_e:
            assert str(log_e) == "log error"


# LLM-generated content at query #21
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True)

def test_log_exception_without_user_msg():
    e = RuntimeError("another test error")
    log_exception(e, force_console=True)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, force_console=True)

def test_log_exception_with_kwargs():
    e = TypeError("type error")
    log_exception(e, timestamp=False, include_proc_id=False, force_console=True)

def test_log_exception_with_logging_failure():
    e = Exception("test exception")
    with patch("flutes.log.log", side_effect=RuntimeError("log failure")):
        with pytest.raises(RuntimeError):
            log_exception(e, force_console=True)


# LLM-generated content at query #22
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert hasattr(test_func, "__wrapped__")


# LLM-generated content at query #24
#--------------------------

```python
def test_skip_keyboard_interrupt():
    skip_exceptions = [BdbQuit]
    assert not any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "main")


# LLM-generated content at query #26
#--------------------------

```python
def test_log_exception_predicate():
    e = ValueError("test error")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #27
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising an exception

def test_log_exception_without_user_msg():
    e = RuntimeError("another test error")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising an exception

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising an exception

def test_log_exception_with_additional_kwargs():
    e = TypeError("type error")
    log_exception(e, force_console=True, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising an exception

def test_log_exception_raises_exception():
    e = Exception("test exception")
    with patch('flutes.log.log', side_effect=Exception("log error")):
        try:
            log_exception(e, timestamp=False, include_proc_id=False)
        except Exception as log_e:
            assert isinstance(log_e, Exception)
            assert str(log_e) == "log error"


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_evaluates_to_false():
    capture_keyboard_interrupt = True
    skip_exceptions = [BdbQuit]
    assert not any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_docstring():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, **kwargs):
        assert one == 1
        assert two == 2
        assert kwargs == {"three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three=3):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_non_matching_args():
    def handler(e, four, **kwargs):
        assert four is None
        assert kwargs == {"one": 1, "two": 2, "three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three=3):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, one=1, two=2, **kwargs):
        assert one == 1
        assert two == 2
        assert kwargs == {"three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three=3):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_varargs_in_handler_raises():
    def handler(e, *args, **kwargs):
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"

def test_exception_wrapper_with_no_exception_arg_raises():
    def handler(**kwargs):
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_kwarg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_kwarg is None
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=None):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three=None, **kw):
        assert one == 1
        assert two == 2
        assert three is None
        assert kw == {"four": 4}

    @exception_wrapper(handler)
    def func(one, two, **kwargs):
        raise ValueError("Test")

    func(1, 2, four=4)

def test_exception_wrapper_with_non_matching_args():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def func(one):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, one=1):
        pass

    try:
        @exception_wrapper(handler)
        def func(one):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #35
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #37
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #39
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_skip_exceptions_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(None)
    def test_func():
        pass
    assert True


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    assert func_with_args(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three=None):
        assert one == 1
        assert two == 2
        assert three is None

    @exception_wrapper(handler)
    def func(one, two, three=None):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["one"] == 1
        assert kw["two"] == 2

    @exception_wrapper(handler)
    def func(**kwargs):
        raise ValueError("test error")

    assert func(one=1, two=2) is None


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    func_with_args(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    next(gen)
    next(gen)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

def test_exception_wrapper_handler_with_defaults():
    def handler_with_defaults(e, arg1, optional_arg=None):
        assert arg1 == 1
        assert optional_arg is None

    @exception_wrapper(handler_with_defaults)
    def func_with_optional(arg1, optional_arg=None):
        raise ValueError("test error")

    func_with_optional(1)

def test_exception_wrapper_handler_var_kw():
    def handler_var_kw(e, arg1, **kw):
        assert arg1 == 1
        assert kw == {"extra": "value"}

    @exception_wrapper(handler_var_kw)
    def func_with_kwargs(arg1, **kwargs):
        raise ValueError("test error")

    func_with_kwargs(1, extra="value")


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def raise_value_error():
        raise ValueError("Test error")

    result = raise_value_error()
    assert result is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def raise_value_error_with_args(arg1, arg2):
        raise ValueError("Test error")

    result = raise_value_error_with_args(1, 2)
    assert result is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_with_error():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = generator_with_error()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, kwarg1, kwarg2):
        assert isinstance(e, ValueError)
        assert kwarg1 == "a"
        assert kwarg2 == "b"

    @exception_wrapper(custom_handler)
    def raise_value_error_with_kwargs(kwarg1, kwarg2):
        raise ValueError("Test error")

    result = raise_value_error_with_kwargs(kwarg1="a", kwarg2="b")
    assert result is None

def test_exception_wrapper_with_varargs():
    def custom_handler(e, var_kw):
        assert isinstance(e, ValueError)
        assert var_kw == {"extra": "value"}

    @exception_wrapper(custom_handler)
    def raise_value_error_with_varargs(**kwargs):
        raise ValueError("Test error")

    result = raise_value_error_with_varargs(extra="value")
    assert result is None


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_that_raises()

def test_exception_wrapper_with_custom_handler():
    handler_called = False

    def custom_handler(e, **kwargs):
        nonlocal handler_called
        handler_called = True
        assert isinstance(e, ValueError)
        assert str(e) == "test error"

    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func_that_raises()
    assert handler_called

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_matching_args():
    def handler_fn(e, arg1, arg2, **kwargs):
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, arg3=3):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func(1, 2, arg3=3)

def test_exception_wrapper_with_no_matching_args():
    def handler_fn(e, arg4=4, **kwargs):
        assert arg4 == 4
        assert kwargs == {"arg1": 1, "arg2": 2}

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        func(1, 2)

def test_exception_wrapper_with_invalid_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=1: None)
        def func(arg1):
            pass


# LLM-generated content at query #46
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_capture_keyboard_interrupt_skip_exceptions():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    gen = test_gen()
    next(gen)
    next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_default_values_in_handler():
    def bad_handler(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func(arg_with_default):
            pass

def test_exception_wrapper_with_varargs_handler():
    def bad_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    def bad_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass


# LLM-generated content at query #48
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == "excepthook"

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == "excepthook"

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_skip_exceptions_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #49
#--------------------------

```python
def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_without_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_always_skips_bdbquit():
    register_ipython_excepthook()
    assert BdbQuit in sys.excepthook.__code__.co_consts


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        return 42

    assert test_func() == 42


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2, optional_arg=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2, optional_arg=None, **kwargs):
        pass

    assert True


# LLM-generated content at query #54
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(None)
    def foo():
        pass
    assert exception_wrapper(None)(foo).__wrapped__ == foo


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_default_values_in_handler():
    def bad_handler(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_varargs_in_handler():
    def bad_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    def bad_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_successful_execution():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_success():
    @exception_wrapper()
    def test_gen():
        yield 1
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(gen)


# LLM-generated content at query #56
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper
        def handler_fn(e, *args):
            pass

        @exception_wrapper(handler_fn)
        def foo():
            pass


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_arg is None
        assert kw == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, arg3=None):
        raise ValueError("test error")

    assert func_with_args(1, 2, arg3=3) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, my_arg=None, **kw):
        assert one == 1
        assert two == 2

    @exception_wrapper(handler)
    def func(one, two, three=None):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_non_matching_args():
    try:
        def handler(e, non_existent_arg):
            pass

        @exception_wrapper(handler)
        def func(one, two):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    try:
        def handler(e, one=1):
            pass

        @exception_wrapper(handler)
        def func(one, two):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #59
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    assert next(gen) is None

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, kwarg1, kwarg2):
        assert isinstance(e, TypeError)
        assert kwarg1 == "a"
        assert kwarg2 == "b"

    @exception_wrapper(custom_handler)
    def test_func(kwarg1, kwarg2):
        raise TypeError("Test error")

    assert test_func(kwarg1="a", kwarg2="b") is None

def test_exception_wrapper_with_var_kw():
    def custom_handler(e, **kwargs):
        assert isinstance(e, RuntimeError)
        assert kwargs == {"extra": "value"}

    @exception_wrapper(custom_handler)
    def test_func(extra):
        raise RuntimeError("Test error")

    assert test_func(extra="value") is None


# LLM-generated content at query #61
#--------------------------

```python
def test_exception_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper
        def handler_with_varargs(e, *args):
            pass


# LLM-generated content at query #62
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2, kwargs)

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise TypeError("Custom error")

    test_func(1, 2, extra="value")

    assert handler_called
    assert isinstance(handler_args[0], TypeError)
    assert handler_args[1] == 1
    assert handler_args[2] == 2
    assert handler_args[3] == {"extra": "value"}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise RuntimeError("Generator error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(RuntimeError):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_defaults_in_handler():
    def bad_handler(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func(arg_with_default):
            pass


# LLM-generated content at query #63
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg3):
        raise ValueError("Test error")

    test_func(1, arg3=3)

def test_exception_wrapper_with_varkw():
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs == {"arg1": 1, "arg2": 2}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)


# LLM-generated content at query #64
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(handler_fn=None)
    def test_func():
        pass

    assert True


# LLM-generated content at query #66
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    assert test_func() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        return "handled"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    assert test_func(1, 2) == "handled"

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1=None):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)


# LLM-generated content at query #67
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("custom error")

    func_with_args(1, 2, extra="value")
    assert handler_called
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["kwargs"] == {"extra": "value"}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise StopIteration("generator error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def handler_with_mismatch(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_mismatch)
        def func():
            pass

def test_exception_wrapper_with_default_values_in_handler():
    def handler_with_defaults(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler_with_defaults)
        def func(arg_with_default):
            pass


# LLM-generated content at query #68
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #69
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, default_arg=None):
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(arg1, arg2, default_arg=10):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def handler(e, **kwargs):
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(arg1=1, arg2=2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    with pytest.raises(subprocess.CalledProcessError):
        test_func()


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_docstring():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #72
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skips_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_handler_has_varkw():
    def handler_fn(e, **kwargs):
        pass

    handler_argspec = inspect.getfullargspec(handler_fn)
    assert handler_argspec.varkw is not None


# LLM-generated content at query #74
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper()
    def foo():
        pass

    assert foo.__wrapped__ is not None


# LLM-generated content at query #75
#--------------------------

```python
def test_exception_wrapper_handler_argspec_args_length():
    assert len(inspect.getfullargspec(_unwrap(handler_fn)).args) > 0


# LLM-generated content at query #76
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_raise():
        raise ValueError("test error")

    assert func_raise() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = None

    def custom_handler(e, arg1, arg2):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = (e, arg1, arg2)

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise RuntimeError("custom error")

    func_with_args("a", "b")
    assert handler_called
    assert isinstance(handler_args[0], RuntimeError)
    assert handler_args[1] == "a"
    assert handler_args[2] == "b"

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise StopIteration("generator error")

    gen = gen_func()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, **kwargs):
        pass

    @exception_wrapper(handler)
    def func(arg1, arg2, arg3):
        pass

    func("a", "b", "c")

def test_exception_wrapper_with_non_matching_args():
    def handler(e, non_existent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def func(arg1):
            pass

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, arg1="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def func(arg1):
            pass

def test_exception_wrapper_with_varargs_in_handler():
    def handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def func(arg1):
            pass

def test_exception_wrapper_with_no_exception_arg():
    def handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def func():
            pass


# LLM-generated content at query #77
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #78
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #79
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #80
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #81
#--------------------------

```python
def test_exception_wrapper_without_handler_fn():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #82
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #83
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, arg2, default_arg=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, default_arg="default"):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_varkw():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2, "extra": "value"}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, extra="value")

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_generator():
        yield 1
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    assert next(gen) == 2

def test_exception_wrapper_with_invalid_handler_no_args():
    def custom_handler():
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"

def test_exception_wrapper_with_invalid_handler_varargs():
    def custom_handler(e, *args):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"

def test_exception_wrapper_with_invalid_handler_arg_mismatch():
    def custom_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(existing_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Argument 'non_existent_arg' in exception handler does not match any argument in wrapped method"

def test_exception_wrapper_with_invalid_handler_default_on_matching_arg():
    def custom_handler(e, existing_arg="default"):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(existing_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Argument 'existing_arg' matches wrapped method argument, thus cannot have default values"


# LLM-generated content at query #84
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #85
#--------------------------

```python
def test_exception_wrapper_docstring():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #86
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #87
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #88
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "CurrentProcessName")


# LLM-generated content at query #89
#--------------------------

```python
def test_exception_wrapper_docstring_predicate():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #90
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert isinstance(e, subprocess.CalledProcessError) and e.output is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = TypeError("test type error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("test runtime error")
    log_exception(e, user_msg="Test", force_console=True, timestamp=True, include_proc_id=False)

def test_log_exception_with_logging_error():
    e = Exception("test exception")
    with patch('flutes.log.log', side_effect=Exception("logging error")):
        log_exception(e, force_console=True, timestamp=False)


# LLM-generated content at query #2
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    assert func_with_args(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert matched_arg == "value"
        assert default_arg is None

    @exception_wrapper(handler)
    def func(matched_arg, other_arg="default"):
        raise ValueError("test error")

    assert func("value") is None

def test_exception_wrapper_with_var_kw_handler():
    def handler(e, **kw):
        assert kw["arg1"] == 1
        assert kw["arg2"] == 2

    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def generator_no_error():
        yield 1
        yield 2

    gen = generator_no_error()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert isinstance(e, subprocess.CalledProcessError) and e.output is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #5
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_none():
    @exception_wrapper(handler_fn=None)
    def test_func():
        pass
    assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_none():
    assert exception_wrapper(handler_fn=None) is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def foo():
        pass

    assert foo.__wrapped__ is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = RuntimeError("test runtime error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error_no_output():
    e = subprocess.CalledProcessError(1, "test_cmd")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = TypeError("test type error")
    log_exception(e, user_msg="Type error occurred", force_console=True, timestamp=True, include_proc_id=False)


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "test")
    e.output = b"test"
    log_exception(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #12
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom message"
    log_exception(e, user_msg)
    assert True

def test_log_exception_without_user_msg():
    e = ValueError("test error")
    log_exception(e)
    assert True

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test_output")
    log_exception(e)
    assert True

def test_log_exception_with_kwargs():
    e = ValueError("test error")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True

def test_log_exception_with_logging_error():
    e = ValueError("test error")
    with unittest.mock.patch('flutes.log.log', side_effect=Exception("log error")):
        with pytest.raises(Exception):
            log_exception(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Test user message"
    log_exception(e, user_msg, force_console=True, timestamp=False, include_proc_id=False)

def test_log_exception_without_user_msg():
    e = RuntimeError("test runtime error")
    log_exception(e, force_console=True, timestamp=False, include_proc_id=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    log_exception(e, force_console=True, timestamp=False, include_proc_id=False)

def test_log_exception_with_additional_kwargs():
    e = TypeError("test type error")
    log_exception(e, user_msg="Test kwargs", force_console=True, timestamp=True, include_proc_id=True)

def test_log_exception_with_logging_failure():
    e = Exception("test exception")
    with patch("flutes.log.log", side_effect=RuntimeError("log failure")):
        try:
            log_exception(e, force_console=True, timestamp=False, include_proc_id=False)
        except RuntimeError:
            pass


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_arg is None
        assert kw == {"arg3": 3, "kwargs": {"arg4": 4}}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, *args, arg3=None, **kwargs):
        raise ValueError("test error")

    assert func_with_args(1, 2, "arg1", "arg2", arg3=3, arg4=4) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_handler_arg_mismatch():
    def handler_with_mismatch(e, nonexistent_arg):
        pass

    try:
        @exception_wrapper(handler_with_mismatch)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_handler_with_defaults():
    def handler_with_defaults(e, arg1="default"):
        pass

    try:
        @exception_wrapper(handler_with_defaults)
        def func(arg1):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "CurrentProcessName")


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == 'excepthook'

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == 'excepthook'

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert KeyboardInterrupt in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_skip_exceptions_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in sys.excepthook.__code__.co_consts


# LLM-generated content at query #18
#--------------------------

```python
def test_register_ipython_excepthook_docstring():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook that launches an interactive IPython session upon uncaught exceptions.")


# LLM-generated content at query #19
#--------------------------

```python
def test_register_ipython_excepthook_docstring_predicate():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook")


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, nonexistent_arg):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1=None):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, arg2, default_arg=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, default_arg=10):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(arg1=1, arg2=2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "Success"

    result = test_func()
    assert result == "Success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_generator():
        yield 1
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    assert next(gen) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorkerProcess")


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_matching_args():
    def handler(e, match_arg, default_arg=None):
        assert match_arg == "test"
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(match_arg, other_arg="default"):
        raise ValueError("Test error")

    test_func("test")

def test_exception_wrapper_no_defaults_on_matching():
    def handler(e, match_arg, default_arg="default"):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def test_func(match_arg):
            pass

def test_exception_wrapper_varargs_not_allowed():
    def handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def test_func():
            pass

def test_exception_wrapper_no_exception_arg():
    def handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(handler)
        def test_func():
            pass


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #27
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #28
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = True
    skip_exceptions = [BdbQuit]
    assert not (not capture_keyboard_interrupt)


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #30
#--------------------------

```python
def test_capture_keyboard_interrupt_false_skips_keyboard_interrupt():
    skip_exceptions = []
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func():
        raise ValueError("test error")

    assert func() is None

def test_exception_wrapper_with_custom_handler():
    def handler_fn(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(handler_fn)
    def func(arg1, arg2, arg3=3):
        raise ValueError("test error")

    assert func(1, 2, arg3=3) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def gen_func():
        yield 1
        raise ValueError("test error")

    gen = gen_func()
    assert next(gen) == 1
    assert next(gen) is None

def test_exception_wrapper_with_custom_handler_and_generator():
    def handler_fn(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(handler_fn)
    def gen_func(arg1, arg2=2):
        yield 1
        raise ValueError("test error")

    gen = gen_func(1, arg2=2)
    assert next(gen) == 1
    assert next(gen) is None

def test_exception_wrapper_with_no_args_handler():
    def handler_fn(e):
        assert isinstance(e, ValueError)

    @exception_wrapper(handler_fn)
    def func():
        raise ValueError("test error")

    assert func() is None

def test_exception_wrapper_with_matching_args_handler():
    def handler_fn(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_default_values_handler():
    def handler_fn(e, arg1=1, arg2=2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(handler_fn)
    def func():
        raise ValueError("test error")

    assert func() is None

def test_exception_wrapper_with_varargs_handler():
    def handler_fn(e, *args):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler_fn)

def test_exception_wrapper_with_no_exception_arg_handler():
    def handler_fn():
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler_fn)

def test_exception_wrapper_with_mismatched_args_handler():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1):
        pass

    with pytest.raises(ValueError):
        func(1)

def test_exception_wrapper_with_default_values_in_matching_args_handler():
    def handler_fn(e, arg1=1):
        pass

    @exception_wrapper(handler_fn)
    def func(arg1):
        pass

    with pytest.raises(ValueError):
        func(1)


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, **kwargs):
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(handler)
    def test_func(arg1, arg2=2):
        raise ValueError("test error")

    test_func(1)

def test_exception_wrapper_with_non_matching_args():
    def handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, arg1=None):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def handler():
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument for the exception object" in str(e)


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_evaluates_to_true():
    capture_keyboard_interrupt = False
    assert not capture_keyboard_interrupt


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def foo():
        pass
    assert foo.__wrapped__ is not None


# LLM-generated content at query #38
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, **kwargs):
        raise RuntimeError("custom error")

    func_with_args(1, 2, extra="value")
    assert handler_called
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["kwargs"] == {"extra": "value"}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise StopIteration("generator error")

    gen = generator_that_raises()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_no_exception():
        return "success"

    assert func_no_exception() == "success"

def test_exception_wrapper_with_mismatched_handler_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def func(arg):
            pass

def test_exception_wrapper_with_default_values_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg=None: None)
        def func(arg):
            pass


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #41
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=2):
        raise ValueError("Test error")

    test_func(1)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert matched_arg == "value"
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(matched_arg, other_arg=None):
        raise ValueError("Test error")

    test_func("value")

def test_exception_wrapper_with_var_kw():
    def handler(e, **kw):
        assert kw["arg1"] == 1
        assert kw["arg2"] == 2

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_nested_exception():
    @exception_wrapper()
    def inner_func():
        raise ValueError("Inner error")

    @exception_wrapper()
    def outer_func():
        inner_func()

    outer_func()


# LLM-generated content at query #42
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_capture_keyboard_interrupt_skip_exceptions():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert not hasattr(test_func, '__wrapped__')


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, TypeError)
        assert arg1 == "test"
        assert kwargs == {"arg2": 42}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise TypeError("Test error")

    test_func("test", arg2=42)

def test_exception_wrapper_with_non_matching_args():
    def custom_handler(e, non_existent_arg, **kwargs):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1="default"):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three=None):
        assert isinstance(e, ValueError)
        assert one == 1
        assert two == 2
        assert three is None

    @exception_wrapper(handler)
    def test_func(one, two, three=None):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_var_kw():
    def handler(e, one, **kw):
        assert isinstance(e, ValueError)
        assert one == 1
        assert kw == {"two": 2, "three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three):
        raise ValueError("Test error")

    assert test_func(1, 2, 3) is None

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_gen():
        yield 1
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    assert func_with_args(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert isinstance(e, ValueError)
        assert matched_arg == "test"
        assert default_arg is None

    @exception_wrapper(handler)
    def func(matched_arg, other_arg, default_arg="default"):
        raise ValueError("test error")

    assert func("test", "other") is None

def test_exception_wrapper_with_varkw():
    def handler(e, captured_arg, **kw):
        assert isinstance(e, ValueError)
        assert captured_arg == "test"
        assert kw == {"other_arg": "value", "extra": "extra_value"}

    @exception_wrapper(handler)
    def func(captured_arg, other_arg, **kwargs):
        raise ValueError("test error")

    assert func("test", other_arg="value", extra="extra_value") is None


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert callable(exception_wrapper)


# LLM-generated content at query #49
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_returns_decorator():
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #53
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #54
#--------------------------

```python
def test_skip_exceptions_contains_keyboard_interrupt():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper_handler_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #56
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "main")


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_handler_with_varargs_raises_value_error():
    def handler_fn(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(handler_fn)


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #59
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not False


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_wrapper_docstring():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #61
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #62
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    list(test_gen())

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, extra_arg=None):
        assert isinstance(e, TypeError)
        assert arg1 == "a"
        assert arg2 == "b"
        assert extra_arg is None

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise TypeError("Test error")

    test_func("a", "b")

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert isinstance(e, RuntimeError)
        assert kw["arg1"] == 1
        assert kw["arg2"] == 2

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise RuntimeError("Test error")

    test_func(arg1=1, arg2=2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()


# LLM-generated content at query #63
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert callable(exception_wrapper)


# LLM-generated content at query #64
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_none():
    @exception_wrapper(handler_fn=None)
    def foo():
        pass
    assert True


# LLM-generated content at query #65
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #66
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(None)
    def foo():
        pass
    assert not hasattr(foo, "__wrapped__")


# LLM-generated content at query #67
#--------------------------

```python
def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #68
#--------------------------

```python
def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #69
#--------------------------

```python
def test_exception_wrapper_with_no_args():
    assert not hasattr(exception_wrapper(), "__wrapped__")


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, matched_arg, **kwargs):
        assert isinstance(e, ValueError)
        assert matched_arg == "test"
        assert kwargs == {"unmatched_arg": "value"}

    @exception_wrapper(custom_handler)
    def test_func(matched_arg, unmatched_arg="value"):
        raise ValueError("Test error")

    test_func("test")

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg_with_default="default", **kwargs):
        assert isinstance(e, ValueError)
        assert arg_with_default == "default"
        assert kwargs == {"func_arg": "value"}

    @exception_wrapper(custom_handler)
    def test_func(func_arg):
        raise ValueError("Test error")

    test_func("value")

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args, **kwargs):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_matching_args():
    def custom_handler(e, non_matching_arg, **kwargs):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(matching_arg):
            pass

def test_exception_wrapper_with_default_values_in_matching_args():
    def custom_handler(e, matching_arg="default", **kwargs):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(matching_arg):
            pass


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "A custom handler function takes the following arguments:" in exception_wrapper.__doc__


# LLM-generated content at query #72
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert hasattr(test_func, "__wrapped__")


# LLM-generated content at query #74
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #75
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "A custom handler function takes the following arguments:" in exception_wrapper.__doc__


# LLM-generated content at query #76
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #77
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #78
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def foo():
        return "bar"

    assert foo() == "bar"


# LLM-generated content at query #79
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #80
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    assert True


# LLM-generated content at query #81
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_kwarg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_kwarg is None
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, *, arg3):
        raise ValueError("test error")

    assert func_with_args(1, 2, arg3=3) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, a, b, **kwargs):
        assert a == 1
        assert b == 2
        assert kwargs == {"c": 3}

    @exception_wrapper(handler)
    def func(a, b, c):
        raise ValueError()

    func(1, 2, 3)

def test_exception_wrapper_with_non_matching_args():
    def handler(e, x, **kwargs):
        pass

    try:
        @exception_wrapper(handler)
        def func(a, b):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, a=1):
        pass

    try:
        @exception_wrapper(handler)
        def func(a):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #82
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    gen = test_gen()
    next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three):
        assert one == 1
        assert two == 2
        assert three == 3

    @exception_wrapper(handler)
    def test_func(one, two, three):
        raise ValueError("Test error")

    test_func(1, 2, 3)

def test_exception_wrapper_with_default_values():
    def handler(e, one, two, three=None, four=None):
        assert one == 1
        assert two == 2
        assert three is None
        assert four is None

    @exception_wrapper(handler)
    def test_func(one, two, three=3, four=4):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def handler(e, one, two, **kw):
        assert one == 1
        assert two == 2
        assert kw == {"three": 3, "four": 4}

    @exception_wrapper(handler)
    def test_func(one, two, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, three=3, four=4)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_gen():
        yield 1
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    assert next(gen) == 2


# LLM-generated content at query #83
#--------------------------

```python
def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #84
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    assert func_with_args(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, default_arg=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(handler)
    def func(arg1, arg2, default_arg="default"):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_varkw():
    def handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2, "extra": "value"}

    @exception_wrapper(handler)
    def func(arg1, arg2, **kwargs):
        raise ValueError("test error")

    assert func(1, 2, extra="value") is None

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def generator_no_error():
        yield 1
        yield 2

    gen = generator_no_error()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #85
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #86
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #87
#--------------------------

```python
def test_exception_wrapper_with_handler_fn():
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    assert True


# LLM-generated content at query #88
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("Test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("Test error")

    assert func_with_args(1, 2) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("Test error")

    gen = generator_that_raises()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, default_arg=None):
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None

    @exception_wrapper(handler)
    def func(arg1, arg2, default_arg=10):
        raise ValueError("Test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["arg1"] == 1
        assert kw["arg2"] == 2

    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("Test error")

    assert func(arg1=1, arg2=2) is None

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def generator_no_error():
        yield 1
        yield 2

    gen = generator_no_error()
    assert next(gen) == 1
    assert next(gen) == 2
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #89
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = RuntimeError("test runtime error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = TypeError("test type error")
    log_exception(e, user_msg="Type error occurred", force_console=True, timestamp=True, include_proc_id=False)

def test_log_exception_raises_another_exception():
    e = Exception("test exception")
    with patch('flutes.log.log', side_effect=RuntimeError("log error")):
        try:
            log_exception(e, force_console=True, timestamp=False)
        except RuntimeError:
            pass


# LLM-generated content at query #90
#--------------------------

```python
def test_log_exception_with_non_subprocess_error():
    e = ValueError("test error")
    log_exception(e)
    assert True


# LLM-generated content at query #91
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert callable(exception_wrapper)


# LLM-generated content at query #92
#--------------------------

```python
def test_exception_wrapper_with_varargs_raises_value_error():
    def handler_fn(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(handler_fn)


# LLM-generated content at query #93
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #94
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_matching_args():
    def handler(e, arg1, arg2, default_arg=None, **kwargs):
        assert arg1 == 1
        assert arg2 == 2
        assert default_arg is None
        assert kwargs == {"arg3": 3}

    @exception_wrapper(handler)
    def test_func(arg1, arg2, arg3=None, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_non_matching_args():
    def handler(e, non_matching_arg=None, **kwargs):
        assert non_matching_arg is None
        assert kwargs == {"arg1": 1, "arg2": 2}

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_subprocess_error():
    import subprocess

    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()


# LLM-generated content at query #95
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, matched_arg, default_arg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert matched_arg == 1
        assert default_arg is None
        assert kwargs == {"unmatched_arg": 2}

    @exception_wrapper(custom_handler)
    def test_func(matched_arg, unmatched_arg):
        raise ValueError("test error")

    test_func(1, unmatched_arg=2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()


# LLM-generated content at query #96
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #97
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass

def test_exception_wrapper_with_defaults_in_handler():
    def custom_handler(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass


# LLM-generated content at query #98
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    list(test_gen())

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert matched_arg == "value"
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(matched_arg, other_arg="default"):
        raise ValueError("Test error")

    test_func("value")

def test_exception_wrapper_with_non_matching_args():
    def handler(e, non_matching_arg):
        assert non_matching_arg is None

    @exception_wrapper(handler)
    def test_func(other_arg):
        raise ValueError("Test error")

    test_func("value")

def test_exception_wrapper_with_var_kw():
    def handler(e, **kw):
        assert kw == {"arg1": 1, "arg2": 2}

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def test_gen():
        yield 1
        yield 2

    assert list(test_gen()) == [1, 2]


# LLM-generated content at query #99
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #100
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    assert not False


# LLM-generated content at query #101
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """


# LLM-generated content at query #102
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_defaults_in_handler():
    def bad_handler(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func(arg_with_default):
            pass

def test_exception_wrapper_with_varargs_handler():
    def bad_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    def bad_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func():
            pass

def test_exception_wrapper_with_kwargs_in_handler():
    def handler(e, **kwargs):
        assert "extra" in kwargs
        assert kwargs["extra"] == "value"

    @exception_wrapper(handler)
    def test_func(**kwargs):
        raise ValueError("Test error")

    test_func(extra="value")

def test_exception_wrapper_with_mixed_args():
    def handler(e, required_arg, optional_arg=None, **kwargs):
        assert required_arg == 1
        assert optional_arg == 2
        assert "extra" in kwargs

    @exception_wrapper(handler)
    def test_func(required_arg, optional_arg=2, **kwargs):
        raise ValueError("Test error")

    test_func(1, extra="value")


# LLM-generated content at query #103
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__code__.co_consts
    assert KeyboardInterrupt in register_ipython_excepthook.__code__.co_consts

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert KeyboardInterrupt in register_ipython_excepthook.__code__.co_consts

def test_register_ipython_excepthook_capture_keyboard_interrupt_no_skip():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in register_ipython_excepthook.__code__.co_consts


# LLM-generated content at query #104
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #105
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_args = {}

    def custom_handler(e, arg1, arg2, default_arg=None, **kwargs):
        nonlocal handler_called, handler_args
        handler_called = True
        handler_args = {"e": e, "arg1": arg1, "arg2": arg2, "default_arg": default_arg, "kwargs": kwargs}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, default_arg="default", **kwargs):
        raise RuntimeError("test error")

    func_with_args(1, 2, extra_kw="extra")
    assert handler_called
    assert isinstance(handler_args["e"], RuntimeError)
    assert handler_args["arg1"] == 1
    assert handler_args["arg2"] == 2
    assert handler_args["default_arg"] == "default"
    assert handler_args["kwargs"] == {"extra_kw": "extra"}

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise StopIteration("test error")

    gen = generator_that_raises()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_invalid_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func():
            pass

    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg=None: None)
        def func(arg):
            pass


# LLM-generated content at query #106
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #107
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


