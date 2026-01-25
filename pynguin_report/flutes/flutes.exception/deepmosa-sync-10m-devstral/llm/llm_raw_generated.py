####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert BdbQuit in register_ipython_excepthook.__code__.co_consts
    assert KeyboardInterrupt in register_ipython_excepthook.__code__.co_consts


# LLM-generated content at query #2
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    assert test_func() is None

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2=2):
        raise ValueError("Test error")

    assert test_func(1) is None

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    gen = test_gen()
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, a, b, c=None):
        assert a == 1
        assert b == 2
        assert c is None

    @exception_wrapper(handler)
    def func(a, b, c=3):
        raise ValueError()

    func(1, 2)

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["x"] == 1
        assert kw["y"] == 2

    @exception_wrapper(handler)
    def func(**kwargs):
        raise ValueError()

    func(x=1, y=2)

def test_exception_wrapper_with_varargs_in_handler_raises():
    def handler(e, *args):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler)

def test_exception_wrapper_with_no_exception_arg_raises():
    def handler():
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler)

def test_exception_wrapper_with_default_values_in_matching_args_raises():
    def handler(e, a=1):
        pass

    def func(a):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler)(func)

def test_exception_wrapper_with_non_matching_args_raises():
    def handler(e, z):
        pass

    def func(a):
        pass

    with pytest.raises(ValueError):
        exception_wrapper(handler)(func)


# LLM-generated content at query #3
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom user message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = TypeError("another test error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "cmd", output="error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error_no_output():
    e = subprocess.CalledProcessError(1, "cmd")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("test runtime error")
    log_exception(e, user_msg="Additional context", force_console=True, timestamp=False, include_proc_id=False)


# LLM-generated content at query #4
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_log_exception_predicate_with_non_called_process_error():
    e = ValueError("test error")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)

def test_log_exception_predicate_with_called_process_error_no_output():
    e = subprocess.CalledProcessError(1, "cmd")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)

def test_log_exception_predicate_with_called_process_error_with_output():
    e = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #6
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
        assert arg1 == "test"
        assert arg2 == 42

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test", 42)

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

def test_exception_wrapper_handler_with_defaults():
    def handler_with_defaults(e, arg1, optional_arg="default"):
        assert arg1 == "test"
        assert optional_arg == "default"

    @exception_wrapper(handler_with_defaults)
    def test_func(arg1):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test")

def test_exception_wrapper_handler_with_kwargs():
    def handler_with_kwargs(e, **kwargs):
        assert kwargs["arg1"] == "test"
        assert kwargs["arg2"] == 42

    @exception_wrapper(handler_with_kwargs)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func("test", 42)


# LLM-generated content at query #7
#--------------------------

```python
def test_log_exception_with_called_process_error_and_output():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


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

    gen = test_gen()
    next(gen)
    next(gen)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_matching_args():
    def custom_handler(e, arg1, arg2=None):
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_kwargs():
    def custom_handler(e, **kwargs):
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)


# LLM-generated content at query #9
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

    test_func(1, 2, arg3=3)

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_subprocess_error():
    import subprocess
    e = subprocess.CalledProcessError(1, "test", output=b"error")
    @exception_wrapper()
    def test_func():
        raise e

    test_func()


# LLM-generated content at query #10
#--------------------------

```python
def test_log_exception_predicate():
    e = Exception("test")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #11
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
    def test_func(arg1, arg2, *, arg3):
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
    def handler(e, a, b, **kwargs):
        assert a == 1
        assert b == 2
        assert kwargs == {"c": 3}

    @exception_wrapper(handler)
    def func(a, b, *, c):
        raise ValueError("Test")

    func(1, 2, c=3)

def test_exception_wrapper_with_non_matching_args():
    def handler(e, x, y, **kwargs):
        pass

    try:
        @exception_wrapper(handler)
        def func(a, b):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

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

def test_exception_wrapper_with_varargs_in_handler():
    def handler(e, *args):
        pass

    try:
        @exception_wrapper(handler)
        def func(a):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def handler():
        pass

    try:
        @exception_wrapper(handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = "error output"
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #13
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert hasattr(inspect, "getfullargspec")


# LLM-generated content at query #14
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
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_handler_with_defaults():
    def custom_handler(e, arg1, arg2=None):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    test_func(1)

def test_exception_wrapper_handler_with_kwargs():
    def custom_handler(e, arg1, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert kwargs == {"arg2": 2}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, arg2=2)


# LLM-generated content at query #15
#--------------------------

```python
def test_register_ipython_excepthook_docstring():
    assert register_ipython_excepthook.__doc__.startswith("Register an exception hook")


# LLM-generated content at query #16
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    log_exception(e)
    assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_log_exception_predicate_true():
    e = subprocess.CalledProcessError(1, "cmd")
    e.output = b"output"
    assert isinstance(e, subprocess.CalledProcessError) and e.output is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert sys.excepthook.__code__.co_name == 'excepthook'

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert sys.excepthook.__code__.co_name == 'excepthook'

def test_register_ipython_excepthook_skip_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    assert sys.excepthook.__code__.co_name == 'excepthook'


# LLM-generated content at query #19
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None
    assert KeyboardInterrupt in register_ipython_excepthook.__defaults__[0]

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None
    assert KeyboardInterrupt not in register_ipython_excepthook.__defaults__[0]


# LLM-generated content at query #20
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

    assert test_func(1, 2) == "handled"

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
    def custom_handler(e, missing_arg):
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


# LLM-generated content at query #21
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #23
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert hasattr(exception_wrapper, "__wrapped__") is False


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #25
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
        assert arg1 == "value1"
        assert arg2 == "value2"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func("value1", "value2")

def test_exception_wrapper_generator():
    @exception_wrapper()
    def test_gen():
        yield 1
        raise ValueError("Test error")

    list(test_gen())

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == "value1"
        assert arg2 == "value2"
        assert kwargs == {"arg3": "value3"}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func("value1", arg2="value2", arg3="value3")

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

    assert list(test_gen()) == [1, 2]


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #27
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    skip_exceptions = [BdbQuit]
    assert not any(BdbQuit is exc_type for exc_type in skip_exceptions) is False


# LLM-generated content at query #28
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    assert not False


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        @exception_wrapper(lambda e, *args: None)
        def foo():
            pass


# LLM-generated content at query #30
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.

    :param capture_keyboard_interrupt: If ``False``, an uncaught :py:exc:`KeyboardInterrupt` exception will not trigger
        the IPython debugger. Defaults to ``False``.
    """


# LLM-generated content at query #31
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #32
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
    def func(matched_arg, other_arg, default_arg="default"):
        raise ValueError("Test error")

    assert func("value", "other") is None

def test_exception_wrapper_with_var_kw():
    def handler(e, **kw):
        assert kw["arg1"] == 1
        assert kw["arg2"] == 2

    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("Test error")

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
def test_register_ipython_excepthook_predicate_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert "PoolWorker" not in mp.current_process().name


# LLM-generated content at query #36
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

    assert test_func(1, 2) == "handled"

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
    def custom_handler(e, non_existent_arg):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
    except ValueError as e:
        assert str(e) == "Argument 'non_existent_arg' in exception handler does not match any argument in wrapped method"

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1=None):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass
    except ValueError as e:
        assert str(e) == "Argument 'arg1' matches wrapped method argument, thus cannot have default values"

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    try:
        @exception_wrapper(custom_handler)
        def test_func():
            pass
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"


# LLM-generated content at query #37
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


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_with_none_handler():
    @exception_wrapper(None)
    def test_func():
        return True

    assert test_func() is True


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_docstring():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #40
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
    def bad_handler(e, missing_arg):
        pass

    @exception_wrapper(bad_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "does not match" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def bad_handler(e, arg1=None):
        pass

    @exception_wrapper(bad_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_handler():
    def bad_handler(e, *args):
        pass

    try:
        exception_wrapper(bad_handler)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def bad_handler():
        pass

    try:
        exception_wrapper(bad_handler)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #41
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
    def handler(e, one, two):
        assert one == 1
        assert two == 2

    @exception_wrapper(handler)
    def test_func(one, two):
        raise ValueError("test error")

    test_func(1, two=2)

def test_exception_wrapper_with_extra_kwargs():
    def handler(e, one, **kw):
        assert one == 1
        assert kw == {"two": 2, "three": 3}

    @exception_wrapper(handler)
    def test_func(one, two, three):
        raise ValueError("test error")

    test_func(1, 2, three=3)

def test_exception_wrapper_with_default_values_in_handler():
    def handler(e, one, two=2):
        assert one == 1
        assert two == 2

    @exception_wrapper(handler)
    def test_func(one, two):
        raise ValueError("test error")

    test_func(1, 2)

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

    assert list(test_gen()) == [1, 2]


# LLM-generated content at query #42
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
        yield 2

    gen = test_generator()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, missing_arg):
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

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    try:
        exception_wrapper(custom_handler)
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg_in_handler():
    def custom_handler():
        pass

    try:
        exception_wrapper(custom_handler)
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #44
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
        assert arg1 == "value1"
        assert arg2 == "value2"
        raise RuntimeError("Handled")

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    with pytest.raises(RuntimeError):
        test_func("value1", "value2")

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

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, non_existent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1="default"):
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


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_custom_handler():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
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
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_handler_with_defaults():
    def custom_handler(e, arg1, arg2=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 is None
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg3=3):
        raise ValueError("Test error")

    test_func(1)

def test_exception_wrapper_mismatched_args():
    def custom_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass

def test_exception_wrapper_handler_with_varargs():
    def custom_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(custom_handler)
        def test_func(arg1):
            pass


# LLM-generated content at query #46
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not False  # Ensures the predicate `not capture_keyboard_interrupt` evaluates to True when `capture_keyboard_interrupt` is False


# LLM-generated content at query #47
#--------------------------

```python
def test_exception_wrapper_with_no_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__name__ == "test_func"


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
        raise RuntimeError("custom error")

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
        raise StopIteration("generator error")

    gen = generator_that_raises()
    assert next(gen) == 1
    assert next(gen) is None

def test_exception_wrapper_with_mismatched_handler_args():
    def bad_handler(e, nonexistent_arg):
        pass

    try:
        @exception_wrapper(bad_handler)
        def func():
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def bad_handler(e, arg_with_default="default"):
        pass

    try:
        @exception_wrapper(bad_handler)
        def func(arg_with_default):
            pass
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot have default values" in str(e)


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_with_handler_fn_none():
    @exception_wrapper(handler_fn=None)
    def foo():
        pass
    assert foo.__wrapped__ is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg=user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = TypeError("another test error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_command", output="error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("kwargs test error")
    log_exception(e, user_msg="Additional kwargs test", force_console=True, timestamp=True, include_proc_id=False)


# LLM-generated content at query #2
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
    def handler(e, x, y, z=3):
        assert isinstance(e, ValueError)
        assert x == 1
        assert y == 2
        assert z == 3

    @exception_wrapper(handler)
    def test_func(x, y, z=3):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert isinstance(e, ValueError)
        assert kw["a"] == 1
        assert kw["b"] == 2

    @exception_wrapper(handler)
    def test_func(a, b):
        raise ValueError("Test error")

    test_func(a=1, b=2)


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_register_ipython_excepthook_predicate_false():
    capture_keyboard_interrupt = True
    skip_exceptions = [BdbQuit]
    assert not (not capture_keyboard_interrupt)


# LLM-generated content at query #5
#--------------------------

```python
def test_skip_keyboard_interrupt():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #6
#--------------------------

```python
def test_exception_wrapper_docstring_predicate():
    assert "Function decorator that calls the specified handler function when a exception occurs inside the decorated" in exception_wrapper.__doc__


# LLM-generated content at query #7
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

def test_log_exception_with_additional_kwargs():
    e = TypeError("type error")
    log_exception(e, timestamp=False, force_console=True)

def test_log_exception_with_logging_failure():
    e = Exception("test error")
    with patch("flutes.log.log", side_effect=Exception("log failure")):
        log_exception(e, force_console=True)


# LLM-generated content at query #8
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

    test_func(1, 2, arg3=3)

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
    def custom_handler(e, non_existent_arg):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
    except ValueError as e:
        assert "does not match any argument in wrapped method" in str(e)

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


# LLM-generated content at query #9
#--------------------------

```python
def test_exception_wrapper_handler_with_varargs():
    with pytest.raises(ValueError, match="Exception handler cannot have a varargs argument"):
        def handler_fn(e, *args):
            pass

        @exception_wrapper(handler_fn)
        def foo():
            pass


# LLM-generated content at query #10
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert not hasattr(test_func, "__wrapped__")


# LLM-generated content at query #11
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #12
#--------------------------

```python
def test_log_exception_predicate_false():
    e = subprocess.CalledProcessError(1, "test")
    e.output = b"test output"
    log_exception(e)
    assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert "Register an exception hook that launches an interactive IPython session upon uncaught exceptions." in register_ipython_excepthook.__doc__


# LLM-generated content at query #14
#--------------------------

```python
def test_log_exception_with_non_called_process_error():
    e = ValueError("test error")
    assert not (isinstance(e, subprocess.CalledProcessError) and e.output is not None)


# LLM-generated content at query #15
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not ("PoolWorker" in "NotAPoolWorker")


# LLM-generated content at query #16
#--------------------------

```python
def test_capture_keyboard_interrupt_false():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    if not capture_keyboard_interrupt:
        skip_exceptions.append(KeyboardInterrupt)
    assert KeyboardInterrupt in skip_exceptions


# LLM-generated content at query #17
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "CurrentProcess")


# LLM-generated content at query #18
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
    e = subprocess.CalledProcessError(1, "test_cmd", output="error output")
    log_exception(e, force_console=True)

def test_log_exception_with_additional_kwargs():
    e = TypeError("type error")
    log_exception(e, user_msg="Type error occurred", timestamp=False, include_proc_id=False, force_console=True)

def test_log_exception_raises_exception():
    e = Exception("original error")
    with patch("flutes.log.log", side_effect=RuntimeError("log error")):
        try:
            log_exception(e, force_console=True)
        except RuntimeError:
            pass


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in mp.current_process().name)


# LLM-generated content at query #20
#--------------------------

```python
def test_exception_wrapper_without_handler():
    assert exception_wrapper() is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #22
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert "PoolWorker" in "PoolWorker-1"


# LLM-generated content at query #23
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
    e = RuntimeError("test runtime error")
    log_exception(e, user_msg="Additional context", force_console=True, timestamp=True, include_proc_id=False)

def test_log_exception_with_non_subprocess_error_and_output():
    e = Exception("generic error")
    e.output = "some output"
    log_exception(e, force_console=True, timestamp=False)


# LLM-generated content at query #24
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert exception_wrapper is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_log_exception_without_user_msg():
    e = ValueError("test error")
    log_exception(e)

def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg)

def test_log_exception_with_kwargs():
    e = ValueError("test error")
    log_exception(e, timestamp=False, include_proc_id=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test_output")
    log_exception(e)

def test_log_exception_with_subprocess_error_and_output():
    e = subprocess.CalledProcessError(1, "test_cmd", output="test_output")
    log_exception(e, user_msg="Custom error message")

def test_log_exception_with_subprocess_error_and_no_output():
    e = subprocess.CalledProcessError(1, "test_cmd")
    log_exception(e, user_msg="Custom error message")


# LLM-generated content at query #26
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def func_that_raises():
        raise ValueError("test error")

    func_that_raises()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == "test"

    @exception_wrapper(custom_handler)
    def func_with_args(arg1, arg2):
        raise ValueError("test error")

    func_with_args(1, "test")

def test_exception_wrapper_with_generator():
    @exception_wrapper()
    def generator_that_raises():
        yield 1
        raise ValueError("test error")
        yield 2

    gen = generator_that_raises()
    assert next(gen) == 1
    next(gen)

def test_exception_wrapper_with_matching_args():
    def handler(e, matched_arg, default_arg=None):
        assert matched_arg == "value"
        assert default_arg is None

    @exception_wrapper(handler)
    def func(matched_arg, other_arg, default_arg="default"):
        raise ValueError("test")

    func("value", "other")

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert kw["arg1"] == 1
        assert kw["arg2"] == "test"

    @exception_wrapper(handler)
    def func(arg1, arg2):
        raise ValueError("test")

    func(1, "test")

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def func_no_error():
        return "success"

    assert func_no_error() == "success"

def test_exception_wrapper_with_subprocess_error():
    @exception_wrapper()
    def func_with_subprocess_error():
        e = subprocess.CalledProcessError(1, "cmd", output=b"error")
        raise e

    func_with_subprocess_error()


# LLM-generated content at query #27
#--------------------------

```python
def test_skip_exceptions_initialization():
    skip_exceptions = [BdbQuit]
    assert skip_exceptions == [BdbQuit]


# LLM-generated content at query #28
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #29
#--------------------------

```python
def test_exception_wrapper_handler_args_with_defaults():
    def handler_fn(e, arg1, arg2=None, **kwargs):
        pass

    @exception_wrapper(handler_fn)
    def test_func(arg1, arg2, arg3=None):
        pass

    assert True


# LLM-generated content at query #30
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

def test_exception_wrapper_with_mismatched_handler_args():
    def custom_handler(e, non_existent_arg):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
    except ValueError as e:
        assert "does not match any argument" in str(e)

def test_exception_wrapper_with_default_values_in_handler():
    def custom_handler(e, arg1=None):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
    except ValueError as e:
        assert "cannot have default values" in str(e)

def test_exception_wrapper_with_varargs_in_handler():
    def custom_handler(e, *args):
        pass

    @exception_wrapper(custom_handler)
    def test_func(arg1):
        raise ValueError("Test error")

    try:
        test_func(1)
    except ValueError as e:
        assert "cannot have a varargs argument" in str(e)

def test_exception_wrapper_with_no_exception_arg():
    def custom_handler():
        pass

    @exception_wrapper(custom_handler)
    def test_func():
        raise ValueError("Test error")

    try:
        test_func()
    except ValueError as e:
        assert "must have a positional argument" in str(e)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_evaluates_to_true():
    capture_keyboard_interrupt = False
    skip_exceptions = [BdbQuit]
    skip_exceptions.append(KeyboardInterrupt)
    assert any(KeyboardInterrupt is exc_type for exc_type in skip_exceptions)


# LLM-generated content at query #32
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #33
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #34
#--------------------------

```python
def test_register_ipython_excepthook_capture_keyboard_interrupt_false():
    assert not capture_keyboard_interrupt


# LLM-generated content at query #35
#--------------------------

```python
def test_log_exception_with_user_msg():
    e = ValueError("test error")
    user_msg = "Custom error message"
    log_exception(e, user_msg, force_console=True, timestamp=False)

def test_log_exception_without_user_msg():
    e = TypeError("type error")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_subprocess_error():
    e = subprocess.CalledProcessError(1, "cmd", output="error output")
    log_exception(e, force_console=True, timestamp=False)

def test_log_exception_with_additional_kwargs():
    e = RuntimeError("runtime error")
    log_exception(e, force_console=True, timestamp=True, include_proc_id=False)

def test_log_exception_raises_another_exception():
    e = Exception("original error")
    with patch("flutes.log.log", side_effect=RuntimeError("log error")):
        try:
            log_exception(e, force_console=True, timestamp=False)
        except RuntimeError:
            pass


# LLM-generated content at query #36
#--------------------------

```python
def test_register_ipython_excepthook_predicate():
    skip_exceptions = [BdbQuit]
    capture_keyboard_interrupt = False
    skip_exceptions.append(KeyboardInterrupt)
    assert not capture_keyboard_interrupt


# LLM-generated content at query #37
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass

    assert test_func.__wrapped__ is not None


# LLM-generated content at query #38
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "current_process_name")


# LLM-generated content at query #39
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    test_func()

def test_exception_wrapper_with_custom_handler():
    def handler(e, arg1, arg2):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2

    @exception_wrapper(handler)
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
    def test_func(matched_arg, other_arg="default"):
        raise ValueError("Test error")

    test_func("test")

def test_exception_wrapper_with_varkw():
    def handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs == {"arg1": 1, "arg2": 2}

    @exception_wrapper(handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_with_subprocess_error():
    import subprocess

    @exception_wrapper()
    def test_func():
        raise subprocess.CalledProcessError(1, "cmd", output=b"error")

    test_func()


# LLM-generated content at query #40
#--------------------------

```python
def test_exception_wrapper_predicate_false():
    assert not hasattr(exception_wrapper, "__wrapped__")


# LLM-generated content at query #41
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
    assert KeyboardInterrupt in sys.excepthook.__code__.co_consts

def test_register_ipython_excepthook_skip_exceptions_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert KeyboardInterrupt not in sys.excepthook.__code__.co_consts


# LLM-generated content at query #42
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

def test_register_ipython_excepthook_skips_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=False)
    assert sys.excepthook is not None
    assert sys.excepthook.__name__ == 'excepthook'


# LLM-generated content at query #43
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert "Function decorator" in exception_wrapper.__doc__


# LLM-generated content at query #44
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"


# LLM-generated content at query #45
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def foo():
        pass
    assert foo.__wrapped__ is not None


# LLM-generated content at query #46
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        pass
    assert exception_wrapper() is not None


# LLM-generated content at query #47
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
    def custom_handler(e, **kwargs):
        assert isinstance(e, ValueError)
        assert kwargs["arg1"] == 1
        assert kwargs["arg2"] == 2

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2):
        raise ValueError("Test error")

    test_func(1, 2)

def test_exception_wrapper_with_mixed_args():
    def custom_handler(e, arg1, arg2, arg_with_default=None, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert arg_with_default is None
        assert kwargs["arg3"] == 3

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3, arg_with_default=None):
        raise ValueError("Test error")

    test_func(1, 2, 3)


# LLM-generated content at query #48
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
    def test_generator():
        yield 1
        raise ValueError("Test error")

    gen = test_generator()
    next(gen)
    next(gen)

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs["kwarg1"] == "value1"

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("Test error")

    test_func(1, 2, kwarg1="value1")

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"

def test_exception_wrapper_invalid_handler_no_exception_arg():
    def invalid_handler():
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(invalid_handler)
        def test_func():
            pass

def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(invalid_handler)
        def test_func():
            pass

def test_exception_wrapper_invalid_handler_mismatched_args():
    def invalid_handler(e, nonexistent_arg):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(invalid_handler)
        def test_func():
            pass

def test_exception_wrapper_invalid_handler_default_values():
    def invalid_handler(e, arg1=None):
        pass

    with pytest.raises(ValueError):
        @exception_wrapper(invalid_handler)
        def test_func(arg1):
            pass


# LLM-generated content at query #49
#--------------------------

```python
def test_exception_wrapper_with_default_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

def test_exception_wrapper_with_custom_handler():
    def custom_handler(e, arg1, arg2, my_arg=None, **kw):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert my_arg is None
        assert kw == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, arg3=3):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func(1, 2, arg3=3)

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


# LLM-generated content at query #50
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #51
#--------------------------

```python
def test_exception_wrapper_predicate():
    assert not ("PoolWorker" in "test_process_name")


# LLM-generated content at query #52
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def test_func():
        raise ValueError("Test error")

    try:
        test_func()
    except ValueError:
        pass


# LLM-generated content at query #53
#--------------------------

```python
def test_register_ipython_excepthook_default():
    register_ipython_excepthook()
    assert sys.excepthook is not None

def test_register_ipython_excepthook_with_keyboard_interrupt():
    register_ipython_excepthook(capture_keyboard_interrupt=True)
    assert sys.excepthook is not None


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
    def handler(e, matched_arg, default_arg=None):
        assert isinstance(e, ValueError)
        assert matched_arg == "test"
        assert default_arg is None

    @exception_wrapper(handler)
    def test_func(matched_arg, other_arg="default"):
        raise ValueError("Test error")

    assert test_func("test") is None

def test_exception_wrapper_with_kwargs():
    def handler(e, **kw):
        assert isinstance(e, ValueError)
        assert kw["key1"] == "value1"
        assert kw["key2"] == "value2"

    @exception_wrapper(handler)
    def test_func(**kwargs):
        raise ValueError("Test error")

    assert test_func(key1="value1", key2="value2") is None


# LLM-generated content at query #55
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
        yield 2

    gen = test_gen()
    assert next(gen) == 1
    try:
        next(gen)
    except StopIteration:
        pass

def test_exception_wrapper_with_kwargs():
    def custom_handler(e, arg1, arg2, **kwargs):
        assert isinstance(e, ValueError)
        assert arg1 == 1
        assert arg2 == 2
        assert kwargs == {"arg3": 3}

    @exception_wrapper(custom_handler)
    def test_func(arg1, arg2, **kwargs):
        raise ValueError("test error")

    test_func(1, 2, arg3=3)

def test_exception_wrapper_no_exception():
    @exception_wrapper()
    def test_func():
        return "success"

    assert test_func() == "success"

def test_exception_wrapper_invalid_handler_no_exception_arg():
    def invalid_handler():
        pass

    try:
        exception_wrapper(invalid_handler)
    except ValueError as e:
        assert str(e) == "Exception handler must have a positional argument for the exception object"

def test_exception_wrapper_invalid_handler_with_varargs():
    def invalid_handler(e, *args):
        pass

    try:
        exception_wrapper(invalid_handler)
    except ValueError as e:
        assert str(e) == "Exception handler cannot have a varargs argument (*args)"

def test_exception_wrapper_invalid_handler_arg_mismatch():
    def invalid_handler(e, nonexistent_arg):
        pass

    @exception_wrapper(invalid_handler)
    def test_func(arg1):
        pass

    try:
        test_func(1)
    except ValueError as e:
        assert str(e) == "Argument 'nonexistent_arg' in exception handler does not match any argument in wrapped method"

def test_exception_wrapper_invalid_handler_default_value():
    def invalid_handler(e, arg1=None):
        pass

    try:
        exception_wrapper(invalid_handler)
    except ValueError as e:
        assert str(e) == "Argument 'arg1' matches wrapped method argument, thus cannot have default values"


# LLM-generated content at query #56
#--------------------------

```python
def test_exception_wrapper_docstring_exists():
    assert exception_wrapper.__doc__ is not None
    assert len(exception_wrapper.__doc__) > 0


# LLM-generated content at query #57
#--------------------------

```python
def test_exception_wrapper_without_handler():
    @exception_wrapper()
    def foo():
        pass

    assert foo.__name__ == "foo"


