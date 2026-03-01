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
    var_0 = True
    module_0.run_command(var_0, verbose=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.CommandResult()

def test_case_5():
    var_0 = False
    var_1 = 'false'
    var_2 = [var_1]
    module_0.run_command(var_2, return_output=var_0, ignore_errors=var_0)

def test_case_6():
    var_0 = 'echo'
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.run_command(var_1, return_output=var_2, ignore_errors=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_4 = module_0.run_command(var_1, verbose=var_2, return_output=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3

def test_case_7():
    var_0 = 'echo'
    var_1 = None
    var_2 = module_0.run_command(var_0, timeout=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_8():
    var_0 = 'echo'
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.run_command(var_1, return_output=var_2, ignore_errors=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_4 = 'P\nf:~_\\q%\r\x0bjB7Hh>\x0cK\n'
    var_5 = False
    var_6 = 'sleep'
    var_7 = module_0.run_command(var_0, return_output=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3
    var_8 = [var_6, var_6, var_6, var_6, var_4, var_0, var_0, var_6, var_4, var_0, var_6, var_4, var_4, var_4, var_0, var_4, var_4, var_4, var_4, var_4, var_6, var_4, var_6, var_0, var_0, var_4, var_4, var_4, var_6, var_4, var_4, var_0, var_6, var_6, var_4, var_0, var_4, var_6, var_0, var_0, var_6, var_4, var_4, var_4, var_4, var_0, var_4, var_4, var_6, var_6, var_6, var_4, var_6, var_0, var_6, var_4, var_4, var_4, var_4, var_4, var_4, var_6, var_0, var_4]
    var_9 = module_0.error_wrapper(var_2)
    assert var_9 is True
    var_10 = module_0.run_command(var_8, return_output=var_9, ignore_errors=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3

def test_case_9():
    var_0 = 'echo'
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.run_command(var_1, return_output=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_4 = 'T\tS^VV>Rpn'
    var_5 = 'Pyf5q<%HjB7Hh>za'
    var_6 = False
    var_7 = 'sleep'
    var_8 = module_0.run_command(var_0, return_output=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = '2'
    var_10 = [var_7, var_0, var_7, var_4, var_7, var_0, var_9, var_4, var_4, var_5, var_7, var_0, var_5]
    var_11 = module_0.run_command(var_10, timeout=var_6, ignore_errors=var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_11) == 3
    var_12 = [var_0]
    var_13 = None
    var_14 = module_0.error_wrapper(var_13)
    var_15 = module_0.run_command(var_12, verbose=var_2, return_output=var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_15) == 3
    module_0.run_command(var_10, env=var_13, timeout=var_14)

def test_case_10():
    var_0 = 'echo'
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.run_command(var_1, return_output=var_2, ignore_errors=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_4 = 'P\nf:~_;q%H\x0bjB7Hh>\x0cKk'
    var_5 = 'sleep'
    var_6 = None
    var_7 = module_0.error_wrapper(var_5)
    assert var_7 == 'sleep'
    var_8 = module_0.error_wrapper(var_6)
    var_9 = module_0.error_wrapper(var_8)
    var_10 = [var_5, var_5, var_5, var_5, var_0, var_5, var_4, var_4, var_4, var_5, var_5, var_4, var_4, var_5, var_5, var_0, var_4, var_5, var_4, var_4, var_0, var_4, var_4, var_5, var_4, var_4, var_5, var_4, var_5, var_4, var_4, var_4, var_4, var_4, var_5, var_0]
    var_11 = module_0.error_wrapper(var_7)
    assert var_11 == 'sleep'
    var_12 = [var_0]
    var_13 = module_0.run_command(var_12, verbose=var_2, return_output=var_8, ignore_errors=var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_13) == 3
    module_0.run_command(var_10, ignore_errors=var_9)