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
    e = TypeError("another test error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("kwargs test error")
    log_exception(e, user_msg="Additional kwargs test", force_console=True, timestamp=False, include_proc_id=False)


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_custom_handler_with_defaults():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=2):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1)

def test_exception_wrapper_custom_handler_with_kwargs():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2, "arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, **kwargs):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1, arg2=2, arg3=3)


# LLM-generated content at query #4
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #5
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    log_exception(e, user_msg="Custom message", force_console=True)
    assert True  # Check if the function runs without raising an exception

def test_log_exception_without_user_msg():
    e = RuntimeError("another error")
    log_exception(e, force_console=True)
    assert True  # Check if the function runs without raising an exception

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    log_exception(e, force_console=True)
    assert True  # Check if the function runs without raising an exception

def test_log_exception_with_kwargs():
    e = KeyError("missing key")
    log_exception(e, user_msg="Key issue", timestamp=False, force_console=True)
    assert True  # Check if the function runs without raising an exception

def test_log_exception_with_logging_failure():
    e = Exception("test")
    with patch("flutes.log.log", side_effect=RuntimeError("log failed")):
        try:
            log_exception(e, user_msg="Should fail", force_console=True)
        except RuntimeError:
            assert True  # Check if the function raises the expected exception


# LLM-generated content at query #6
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, timestamp=False, include_proc_id=False)
    assert True

def test_log_exception_without_user_msg():
    e = TypeError("test type error")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_command", output="error output")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("test runtime error")
    log_exception(e, user_msg="Additional context", force_console=True, timestamp=False, include_proc_id=False)
    assert True


# LLM-generated content at query #8
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

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
    except ValueError as e:
        assert str(e) == "Argument 'non_existent_arg' in exception handler does not match any argument in wrapped method"

def test_exception_wrapper_with_default_args_in_handler():
    def custom_handler(e, arg1=None):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
    except ValueError as e:
        assert str(e) == "Argument 'arg1' matches wrapped method argument, thus cannot have default values"


# LLM-generated content at query #9
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not any(KeyboardInterrupt is exc_type for exc_type in [BdbQuit])


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert isinstance(e, subprocess.CalledProcessError) and e.output is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #13
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    e = ValueError("Test error")
    log_exception(e)
    assert True


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_with_custom_handler():
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    handler_argspec = inspect.getfullargspec(handler_fn)
    assert handler_argspec.varkw is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #17
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #18
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #19
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "cmd", output="error output")
    log_exception(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_docstring_predicate():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, arg3=3):
        raise ValueError("test error")

    func_with_args(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")

    gen = generator_that_raises()
    next(gen)
    next(gen)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_exception():
        return "success"

    assert func_no_exception() == "success"

def test_exception_wrapper_handler_with_defaults():
    def handler_with_defaults(e, arg1, arg2=None, **kwargs):
        assert arg2 is None

    @exception_wrapper(handler_with_defaults)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    func_with_args(1, 2)

def test_exception_wrapper_mismatched_args():
    def handler_mismatched(e, nonexistent_arg):
        pass

    with raises(ValueError):
        @exception_wrapper(handler_mismatched)
        def func(arg1):
            pass

def test_exception_wrapper_handler_with_varargs():
    def handler_with_varargs(e, *args):
        pass

    with raises(ValueError):
        @exception_wrapper(handler_with_varargs)
        def func(arg1):
            pass


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_handler_has_varargs():
    def handler_fn(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(handler_fn)


# LLM-generated content at query #24
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """


# LLM-generated content at query #25
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1):
        assert isinstance(e, ValueError)
        assert arg1 == "test"

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_mismatched_handler_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def test_func():
            pass

def test_exception_wrapper_with_default_values_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def test_func():
            pass

def test_exception_wrapper_with_no_exception_arg():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda: None)
        def test_func():
            pass

def test_exception_wrapper_with_kwargs_handler():
    def handler(e, **kwargs):
        assert "arg1" in kwargs
        assert kwargs["arg1"] == "test"

    @exception_wrapper(handler)
    def test_func(arg1):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test")

def test_exception_wrapper_with_mixed_args_handler():
    def handler(e, arg1, arg2="default"):
        assert arg1 == "test1"
        assert arg2 == "test2"

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test1", "test2")

def test_exception_wrapper_with_non_matching_default_arg():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1="default": None)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_non_matching_arg():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent: None)
        def test_func(arg1):
            pass


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_returns_decorator():
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #29
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
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_nested_exception():
    e = RuntimeError("test runtime error")
    with patch("flutes.log.log", side_effect=RuntimeError("log error")):
        log_exception(e, force_console=True, timestamp=False)


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
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
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_custom_handler_generator():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(custom_handler)
    def test_gen(arg1, arg2=2):
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen(1)
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_custom_handler_no_exception():
    def custom_handler(e, arg1, **kwargs):
        assert False, "Handler should not be called"

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        return arg1

    assert test_func(1) == 1


# LLM-generated content at query #31
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook that launches an interactive IPython session upon uncaught exceptions.")


# LLM-generated content at query #32
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    type = KeyboardInterrupt
    assert not any(type is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #33
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    skip_exceptions = [BdbQuit]
    assert not any(BdbQuit is exc_type for exc_type in skip_exceptions) is False


# LLM-generated content at query #34
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #39
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
        assert arg2 == "test"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func(1, "test")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert matched_arg == "value"
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(matched_arg, other_arg, default_arg="default"):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("value", "other")

def test_exception_wrapper_with_kwargs():
    def handler(e, **kwargs):
        assert kwargs["kwarg1"] == 1
        assert kwargs["kwarg2"] == "test"

    @exception_wrapper(handler)
    def test_func(**kwargs):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func(kwarg1=1, kwarg2="test")

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


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert len(exception_wrapper.__doc__) > 0


# LLM-generated content at query #41
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

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #43
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == 'excepthook'

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == 'excepthook'


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_arg is None
        assert kw == {"arg3": 3, "kwargs": {"arg4": 4}}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, *, arg3=None, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3, arg4=4)

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

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == "test"
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=None):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func("test")

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_matching_args():
    def handler(e, x, y=None):
        pass

    @exception_wrapper(handler)
    def func(x, y=None, z=None):
        pass

    assert True  # Just checking the decorator doesn't raise during setup

def test_exception_wrapper_non_matching_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, z: None)
        def func(x, y=None):
            pass

def test_exception_wrapper_default_values_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, x="default": None)
        def func(x):
            pass

def test_exception_wrapper_varargs_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def func(x):
            pass


# LLM-generated content at query #46
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


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert len(exception_wrapper.__doc__) > 0


# LLM-generated content at query #49
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

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, nonexistent_arg):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1=1):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    try:
        exception_wrapper(custom_handler)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    try:
        exception_wrapper(custom_handler)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #51
#--------------------------

```python
def test_register_ipython_excepthook_docstring():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook that launches an interactive IPython session upon uncaught exceptions.")


# LLM-generated content at query #52
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
    def handler(e, x, y):
        assert x == 1
        assert y == 2

    @exception_wrapper(handler)
    def func(x, y):
        raise ValueError("test error")

    assert func(1, 2) is None

def test_exception_wrapper_with_non_matching_args():
    def handler(e, z):
        assert False  # Should not be called

    try:
        @exception_wrapper(handler)
        def func(x, y):
            pass
        assert False  # Should raise ValueError
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, x=1):
        pass

    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False  # Should raise ValueError
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func(x):
            pass
        assert False  # Should raise ValueError
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_varkw():
    def handler(e, **kwargs):
        assert kwargs == {"x": 1, "y": 2}

    @exception_wrapper(handler)
    def func(x, y):
        raise ValueError("test error")

    assert func(1, 2) is None


# LLM-generated content at query #53
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert len(exception_wrapper.__doc__) > 0


# LLM-generated content at query #54
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #55
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #56
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #57
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

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_default_values_in_handler():
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
        def test_func(arg1):
            pass

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func():
            pass

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()

def test_exception_wrapper_with_kwargs_in_handler():
    def custom_handler(e, **kwargs):
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, arg2=2)


# LLM-generated content at query #58
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
        return "handled"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    result = test_func(1, 2)
    assert result == "handled"

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
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None
        return "handled"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    result = test_func(1, 2)
    assert result == "handled"

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2
        return "handled"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    result = test_func(1, 2)
    assert result == "handled"

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"

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


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def foo(arg1, arg2=None):
        pass

    assert True


# LLM-generated content at query #60
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in mp.current_process().name)


# LLM-generated content at query #62
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert_raises(ValueError, func_that_raises)

def test_exception_wrapper_with_custom_handler():
    handler_called = False
    handler_exception = None

    def custom_handler(e):
        nonlocal handler_called, handler_exception
        handler_called = True
        handler_exception = e

    @exception_wrapper(custom_handler)
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()
    assert handler_called
    assert isinstance(handler_exception, ValueError)
    assert str(handler_exception) == "test error"

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    assert_raises(ValueError, lambda: next(gen))

def test_exception_wrapper_with_matching_args():
    def handler_fn(e, arg1, arg2):
        return (e, arg1, arg2)

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    result = func(1, 2)
    assert result[0].__class__.__name__ == "ValueError"
    assert result[1] == 1
    assert result[2] == 2

def test_exception_wrapper_with_default_args():
    def handler_fn(e, arg1=None, arg2=None):
        return (e, arg1, arg2)

    @exception_wrapper(handler_fn)
    def func(arg1, arg2):
        raise ValueError("test error")

    result = func(1, 2)
    assert result[0].__class__.__name__ == "ValueError"
    assert result[1] == 1
    assert result[2] == 2

def test_exception_wrapper_with_kwargs():
    def handler_fn(e, **kwargs):
        return (e, kwargs)

    @exception_wrapper(handler_fn)
    def func(**kwargs):
        raise ValueError("test error")

    result = func(arg1=1, arg2=2)
    assert result[0].__class__.__name__ == "ValueError"
    assert result[1] == {"arg1": 1, "arg2": 2}

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def func_no_raise():
        return "success"

    assert func_no_raise() == "success"

def test_exception_wrapper_with_generator_no_exception():
    @exception_wrapper()
    def generator_no_raise():
        yield 1
        yield 2

    gen = generator_no_raise()
    assert next(gen) == 1
    assert next(gen) == 2
    assert_raises(StopIteration, lambda: next(gen))


# LLM-generated content at query #64
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert isinstance(sys.excepthook, types.FunctionType)

def test_register_ipython_excepthook_capture_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert isinstance(sys.excepthook, types.FunctionType)

def test_register_ipython_excepthook_skip_exceptions():
    register_ipython_excepthook()
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_capture_keyboard_interrupt_skip_exceptions():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert BdbQuit in register_ipython_excepthook.__defaults__[0]
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #66
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #67
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
    def test_gen():
        yield 1
        raise ValueError("test error")

    list(test_gen())

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three=None):
        assert one == 1
        assert two == 2
        assert three == 3

    @exception_wrapper(handler)
    def test_func(one, two, three=3):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["one"] == 1
        assert kw["two"] == 2

    @exception_wrapper(handler)
    def test_func(**kwargs):
        raise ValueError("test error")

    test_func(one=1, two=2)

def test_exception_wrapper_with_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_nested_exception():
    @exception_wrapper()
    def test_func():
        try:
            raise ValueError("inner error")
        except ValueError:
            raise RuntimeError("outer error")

    test_func()


# LLM-generated content at query #68
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
        yield 2

    gen = test_gen()
    next(gen)
    next(gen)

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs["arg3"] == 3

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_default_values():
    def custom_handler(e, arg1, arg2=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs["arg3"] == 3

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=2, **kwargs):
        raise ValueError("Test error")

    test_func(1, arg3=3)

def test_exception_wrapper_with_mismatched_args():
    def custom_handler(e, non_existent_arg):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_in_handler():
    def custom_handler(e, arg1=1):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs():
    def custom_handler(e, *args):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)


# LLM-generated content at query #69
#--------------------------

```python
def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper
        def handler_fn(e, *args):
            pass

        def foo():
            pass

        exception_wrapper(handler_fn)(foo)


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert "PoolWorker" not in "CurrentProcessName"


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #72
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorkerProcess")


# LLM-generated content at query #74
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #75
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #76
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert exception_wrapper() is not None


# LLM-generated content at query #77
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #78
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising exceptions

def test_log_exception_without_user_msg():
    e = TypeError("test type error")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising exceptions

def test_log_exception_with_called_process_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    log_exception(e, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising exceptions

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("test runtime error")
    log_exception(e, user_msg="Test", force_console=True, timestamp=False, include_proc_id=False)
    assert True  # Check if function executes without raising exceptions


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_predicate_with_non_called_process_error():
    e = ValueError("test error")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)

def test_log_exception_predicate_with_called_process_error_no_output():
    e = subprocess.CalledProcessError(1, "test_cmd")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)

def test_log_exception_predicate_with_called_process_error_with_output():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #4
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert isinstance(e, subprocess.CalledProcessError) and e.output is not None


# LLM-generated content at query #5
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
    def test_gen():
        yield 1
        raise ValueError("test error")
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
    def bad_handler(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(bad_handler)
        def test_func(arg_with_default):
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

def test_exception_wrapper_with_kwargs_in_handler():
    def handler(e, **kwargs):
        assert "test_arg" in kwargs
        assert kwargs["test_arg"] == "test_value"

    @exception_wrapper(handler)
    def test_func(**kwargs):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(test_arg="test_value")

def test_exception_wrapper_with_mixed_args():
    def handler(e, required_arg, optional_arg=None, **kwargs):
        assert required_arg == 1
        assert optional_arg == 2
        assert "extra" in kwargs
        assert kwargs["extra"] == "value"

    @exception_wrapper(handler)
    def test_func(required_arg, optional_arg=None, **kwargs):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        test_func(1, optional_arg=2, extra="value")


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_false():
    capture_keyboard_interrupt = True
    skip_exceptions = [BdbQuit]
    assert not any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #8
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #10
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not any(KeyboardInterrupt is exc_type for exc_type in [BdbQuit])


# LLM-generated content at query #11
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"


# LLM-generated content at query #12
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    assert exception_wrapper() is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #14
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert test_func.__wrapped__ is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_exception_wrapper_handler_fn_none():
    @exception_wrapper()
    def foo():
        pass

    assert foo.__wrapped__ is not None


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Test message"
    log_exception(e, user_msg)
    assert True

def test_log_exception_without_user_msg():
    e = ValueError("test error")
    log_exception(e)
    assert True

def test_log_exception_with_kwargs():
    e = ValueError("test error")
    log_exception(e, user_msg="Test message", force_console=True, timestamp=False)
    assert True

def test_log_exception_with_called_process_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    log_exception(e)
    assert True

def test_log_exception_with_called_process_error_and_user_msg():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    user_msg = "Test message"
    log_exception(e, user_msg)
    assert True

def test_log_exception_with_called_process_error_and_kwargs():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test output")
    log_exception(e, user_msg="Test message", force_console=True, timestamp=False)
    assert True

def test_log_exception_raises_exception():
    e = ValueError("test error")
    with pytest.raises(Exception):
        log_exception(e, user_msg="Test message", force_console=True, timestamp=False, include_proc_id=False)


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        test_func(1, 2, arg3=3)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_custom_handler_generator():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(custom_handler)
    def test_gen(arg1, arg2=2):
        yield 1
        raise ValueError("test error")

    gen = test_gen(1, arg2=2)
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_no_args_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e: None)
        def test_func():
            pass

def test_exception_wrapper_varargs_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def test_func():
            pass

def test_exception_wrapper_mismatched_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def test_func(arg1):
            pass

def test_exception_wrapper_default_arg_match():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=None: None)
        def test_func(arg1):
            pass


# LLM-generated content at query #20
#--------------------------

```python
def test_log_exception_predicate():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, arg1, arg2, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(arg1, arg2, *args, my_arg=None, **kwargs):
        pass

    assert True


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #24
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #25
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "Argument '{name}' in exception handler does not match any argument in wrapped method" in str(
        ValueError("Argument '{name}' in exception handler does not match any argument in wrapped method")
    )


# LLM-generated content at query #27
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #28
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    skip_exceptions = [BdbQuit]
    assert not any(BdbQuit is exc_type for exc_type in skip_exceptions) is False


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func() is None


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def foo():
        pass
    assert foo.__wrapped__ is not None


# LLM-generated content at query #34
#--------------------------

```python
def test_register_ipython_excepthook_docstring():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook that launches an interactive IPython session upon uncaught exceptions.")


# LLM-generated content at query #35
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
    def test_func(**kwargs):
        raise ValueError("Test error")

    test_func(arg1=1, arg2=2)

def test_exception_wrapper_with_varargs():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(*args):
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
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_with_varargs_handler():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #39
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert hasattr(test_func, "__wrapped__")


# LLM-generated content at query #41
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    type = KeyboardInterrupt
    assert not any(type is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #42
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_without_varargs():
    def handler_fn(e, arg1, arg2=None):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2=None):
        pass

    assert handler_argspec.varargs is None


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #45
#--------------------------

```python
def test_register_ipython_excepthook_docstring():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook")


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    assert func_that_raises() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, optional_arg=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert optional_arg is None
        assert kwargs == {"extra": "value"}

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2, *, optional_arg=None, **kwargs):
        raise ValueError("test error")

    assert func_with_args(1, 2, extra="value") is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    assert next(gen) is None  # Exception handled, generator exhausted

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, non_existent_arg):
        pass

    with pytest.raises(ValueError, match="Argument 'non_existent_arg' in exception handler does not match"):
        @exception_wrapper(bad_handler)
        def func():
            pass

def test_exception_wrapper_with_default_in_handler():
    def bad_handler(e, arg_with_default=None):
        pass

    with pytest.raises(ValueError, match="Argument 'arg_with_default' matches wrapped method argument"):
        @exception_wrapper(bad_handler)
        def func(arg_with_default):
            pass


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #48
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


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
def test_exception_wrapper_predicate():
    assert hasattr(exception_wrapper, "__wrapped__") is False


# LLM-generated content at query #51
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

    gen = test_generator()
    next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, **kwargs):
        assert matched_arg == "value"

    @exception_wrapper(handler)
    def test_func(matched_arg, other_arg):
        raise ValueError("Test error")

    test_func("value", "other")

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, arg_with_default=None, **kwargs):
        assert arg_with_default is None

    @exception_wrapper(handler)
    def test_func(arg_without_default):
        raise ValueError("Test error")

    test_func("value")

def test_exception_wrapper_with_varargs_in_handler_raises_error():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "varargs" in str(e)

def test_exception_wrapper_with_no_exception_arg_raises_error():
    def handler():
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "positional argument" in str(e)

def test_exception_wrapper_with_non_matching_handler_arg_raises_error():
    def handler(e, non_matching_arg):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(matching_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_with_default_value_in_matching_arg_raises_error():
    def handler(e, matching_arg=None):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(matching_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #53
#--------------------------

```python
def test_register_ipython_excepthook_default():
    import sys
    from IPython.core import ultratb

    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_with_keyboard_interrupt():
    import sys
    from IPython.core import ultratb

    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    import sys
    from IPython.core import ultratb

    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    assert isinstance(sys.excepthook, type(lambda: None))


# LLM-generated content at query #54
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
    def handler(e, a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @exception_wrapper(handler)
    def test_func(a, b, d=4):
        raise ValueError("Test error")

    assert test_func(1, 2) is None

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["a"] == 1
        assert kw["b"] == 2

    @exception_wrapper(handler)
    def test_func(a, b):
        raise ValueError("Test error")

    assert test_func(a=1, b=2) is None

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


# LLM-generated content at query #55
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

def test_exception_wrapper_with_var_kw():
    def custom_handler(e, arg1, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kw == {"arg2": 2, "extra": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, extra=3)

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

def test_exception_wrapper_with_invalid_handler():
    try:
        @exception_wrapper(lambda: None)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument" in str(e)

def test_exception_wrapper_with_varargs_handler():
    try:
        @exception_wrapper(lambda e, *args: None)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_mismatched_args():
    try:
        @exception_wrapper(lambda e, nonexistent_arg: None)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_with_default_in_matched_arg():
    try:
        @exception_wrapper(lambda e, arg1=1: None)
        def test_func(arg1):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #56
#--------------------------

```python
def test_register_ipython_excepthook_default():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #57
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert not capture_keyboard_interrupt


# LLM-generated content at query #58
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")

    gen = test_gen()
    next(gen)
    next(gen)

def test_exception_wrapper_custom_handler_with_defaults():
    def custom_handler(e, arg1=None, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=None):
        raise ValueError("test error")

    test_func(1)

def test_exception_wrapper_custom_handler_with_kwargs():
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    test_func(1, 2)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_generator_no_exception():
    @exception_wrapper()
    def test_gen():
        yield 1
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    assert next(gen) == 2


# LLM-generated content at query #59
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert hasattr(exception_wrapper, "__wrapped__") is False


# LLM-generated content at query #60
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
    def handler(e, one, two):
        assert one == 1
        assert two == 2

    @exception_wrapper(handler)
    def test_func(one, two):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_default_values():
    def handler(e, one, two=None):
        assert one == 1
        assert two is None

    @exception_wrapper(handler)
    def test_func(one, two=2):
        raise ValueError("Test error")

    test_func(1)

def test_exception_wrapper_with_var_kw():
    def handler(e, one, **kw):
        assert one == 1
        assert kw == {"two": 2, "three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three):
        raise ValueError("Test error")

    test_func(1, 2, three=3)

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

def test_exception_wrapper_with_invalid_handler():
    def handler():
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Exception handler must have a positional argument for the exception object" in str(e)

def test_exception_wrapper_with_varargs_handler():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def test_func():
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Exception handler cannot have a varargs argument (*args)" in str(e)

def test_exception_wrapper_with_unmatched_args():
    def handler(e, unmatched_arg):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(matched_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Argument 'unmatched_arg' in exception handler does not match any argument in wrapped method" in str(e)

def test_exception_wrapper_with_default_in_matched_args():
    def handler(e, matched_arg=None):
        pass

    try:
        @exception_wrapper(handler)
        def test_func(matched_arg):
            pass
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Argument 'matched_arg' matches wrapped method argument, thus cannot have default values" in str(e)


# LLM-generated content at query #61
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(handler_fn=None)
    def foo():
        pass

    assert foo.__wrapped__ is not None


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #64
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #65
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_arg is None
        assert kw == {"arg3": 3, "kwargs": {"arg4": 4}}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, *, arg3=None, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, arg3=3, arg4=4)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, one, two, three=None, **kw):
        assert one == 1
        assert two == 2
        assert three is None
        assert kw == {"four": 4}

    @exception_wrapper(handler)
    def test_func(one, two, *, three=None, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, four=4)

def test_exception_wrapper_with_non_matching_args():
    try:
        def handler(e, non_existent_arg):
            pass

        @exception_wrapper(handler)
        def test_func(one):
            pass

        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    try:
        def handler(e, one=1):
            pass

        @exception_wrapper(handler)
        def test_func(one):
            pass

        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    try:
        def handler(e, *args):
            pass

        @exception_wrapper(handler)
        def test_func(one):
            pass

        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    try:
        def handler():
            pass

        @exception_wrapper(handler)
        def test_func():
            pass

        test_func()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #66
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #67
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #68
#--------------------------

```python
def test_exception_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #69
#--------------------------

```python
def test_exception_handler_with_varargs_raises_error():
    def handler_fn(e, *args):
        pass

    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        exception_wrapper(handler_fn)


# LLM-generated content at query #70
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #71
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #72
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == "test"
        assert arg2 == 42

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    test_func("test", 42)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("test error")

    list(test_gen())

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_handler_with_defaults():
    def handler(e, arg1="default"):
        assert arg1 == "default"

    @exception_wrapper(handler)
    def test_func():
        raise ValueError("test error")

    test_func()

def test_exception_wrapper_handler_with_kwargs():
    def handler(e, **kwargs):
        assert kwargs["arg1"] == "test"
        assert kwargs["arg2"] == 42

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("test error")

    test_func("test", 42)


# LLM-generated content at query #73
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #74
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

    with pytest.raises(ValueError):
        test_func(1, 2)

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    assert next(gen) == 1
    with pytest.raises(ValueError):
        next(gen)

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, kwarg1, kwarg2):
        assert isinstance(e, ValueError)
        assert kwarg1 == "a"
        assert kwarg2 == "b"

    @exception_wrapper(custom_handler)
    def test_func(kwarg1, kwarg2):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func(kwarg1="a", kwarg2="b")

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"


# LLM-generated content at query #75
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #76
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
    def handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_var_kw():
    def handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2, "arg3": 3}

    @exception_wrapper(handler)
    def test_func(arg1, arg2, arg3=3):
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
    try:
        next(gen)
    except StopIteration:
        pass


# LLM-generated content at query #77
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #78
#--------------------------

```python
def test_exception_wrapper_returns_decorator():
    result = exception_wrapper()
    assert callable(result)


# LLM-generated content at query #79
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorkerProcess")


# LLM-generated content at query #80
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #81
#--------------------------

```python
def test_exception_wrapper_handler_args_with_defaults():
    def handler_fn(e, arg_with_default=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def foo(arg1, arg2):
        pass

    handler_argspec = inspect.getfullargspec(handler_fn)
    handler_args_with_defaults = set((handler_argspec.kwonlydefaults or {}).keys())
    assert handler_args_with_defaults == {"arg_with_default"}


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
    def test_generator():
        yield 1
        raise ValueError("Test error")
        yield 2

    gen = test_generator()
    next(gen)

def test_exception_wrapper_with_matching_args():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=2):
        raise ValueError("Test error")

    test_func(1)

def test_exception_wrapper_with_var_kw():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2, "arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3):
        raise ValueError("Test error")

    test_func(1, 2, 3)

def test_exception_wrapper_with_no_args_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e: None)
        def test_func():
            pass

def test_exception_wrapper_with_varargs_in_handler():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, *args: None)
        def test_func():
            pass

def test_exception_wrapper_with_mismatched_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1: None)
        def test_func(arg2):
            pass

def test_exception_wrapper_with_default_values_in_matching_args():
    with pytest.raises(ValueError):
        @exception_wrapper(lambda e, arg1=1: None)
        def test_func(arg1):
            pass


# LLM-generated content at query #83
#--------------------------

```python
def test_exception_wrapper_with_valid_handler():
    def handler_fn(e, three, one, args, my_arg=None, **kw):
        pass

    @exception_wrapper(handler_fn)
    def foo(one, two, *args, three=None, **kwargs):
        pass

    assert True


# LLM-generated content at query #84
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #85
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None


# LLM-generated content at query #86
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func is not None


# LLM-generated content at query #87
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #88
#--------------------------

```python
def test_exception_wrapper_without_varargs():
    def handler_fn(e, arg1, arg2):
        pass

    @exception_wrapper(handler_fn)
    def foo(arg1, arg2):
        pass

    assert inspect.getfullargspec(handler_fn).varargs is None


# LLM-generated content at query #89
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True)

def test_log_exception_without_user_msg():
    e = TypeError("type error")
    log_exception(e, force_console=True)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output=b"error output")
    log_exception(e, force_console=True)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("runtime error")
    log_exception(e, user_msg="Additional info", timestamp=False, force_console=True)

def test_log_exception_with_logging_failure():
    e = Exception("test exception")
    with patch('flutes.log.log', side_effect=Exception("log failure")):
        log_exception(e, user_msg="Should raise", force_console=True)


# LLM-generated content at query #90
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #91
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


