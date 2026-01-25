# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.run as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.error_wrapper(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.run_command(var_0, env=var_0, timeout=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    var_1 = '\x0c'
    var_2 = True
    module_0.run_command(var_1, cwd=var_0, verbose=var_0, ignore_errors=var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.CommandResult()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = True
    module_0.run_command(var_0, cwd=var_0, verbose=var_1, ignore_errors=var_0)

def test_case_5():
    var_0 = 'pwd'
    var_1 = module_0.run_command(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_6():
    var_0 = 'ls'
    var_1 = True
    var_2 = module_0.error_wrapper(var_0)
    assert var_2 == 'ls'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    var_3 = module_0.run_command(var_2, return_output=var_2, ignore_errors=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_7():
    var_0 = None
    var_1 = module_0.error_wrapper(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    var_2 = 'ls'
    var_3 = module_0.error_wrapper(var_1)
    var_4 = module_0.error_wrapper(var_2)
    assert var_4 == 'ls'
    var_5 = module_0.run_command(var_4, env=var_1, return_output=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_5) == 3
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_6 = -3377.619291664062
    var_7 = module_0.run_command(var_2, timeout=var_6, verbose=var_0, ignore_errors=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3
    var_8 = module_0.run_command(var_4, env=var_0, cwd=var_3, timeout=var_3, verbose=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = module_0.run_command(var_4, env=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_9) == 3

def test_case_8():
    var_0 = 'false'
    var_1 = [var_0]
    var_2 = {}
    module_0.run_command(var_1, **var_2)

def test_case_9():
    var_0 = 'sleep'
    var_1 = '2'
    var_2 = [var_0, var_1]
    var_3 = 0.1
    var_4 = True
    var_5 = {}
    var_6 = module_0.run_command(var_2, timeout=var_3, ignore_errors=var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_7 = var_6.return_code
    assert var_7 == -32768
    var_8 = bool(var_6.captured_output is not None)
    assert var_8 is True

def test_case_10():
    var_0 = 'sleep'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = 0.1
    var_4 = True
    var_5 = {}
    var_6 = module_0.run_command(var_2, timeout=var_3, ignore_errors=var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_7 = var_6.return_code
    assert var_7 == -32768
    with pytest.raises(AttributeError):
        var_8 = bool(var_7.captured_output is not None)
    assert var_8 is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'x'
    var_1 = 10000
    var_2 = var_0 * var_1
    var_3 = 'python3'
    var_4 = f"print('{var_2}')"
    var_5 = [var_3, var_2, var_4]
    var_6 = True
    var_7 = {}
    var_8 = module_0.run_command(var_5, return_output=var_6, ignore_errors=var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_9 = True
    module_0.run_command(var_3, ignore_errors=var_9, **var_4)