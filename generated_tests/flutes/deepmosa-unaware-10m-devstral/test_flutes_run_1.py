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
    module_0.run_command(var_0, env=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    var_1 = None
    var_2 = False
    module_0.run_command(var_0, env=var_1, cwd=var_0, verbose=var_2)

def test_case_3():
    var_0 = 'echo'
    var_1 = module_0.run_command(var_0, verbose=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.CommandResult()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'v|w3aIud4feN\nC.@['
    var_1 = module_0.error_wrapper(var_0)
    assert var_1 == 'v|w3aIud4feN\nC.@['
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    module_0.run_command(var_0, cwd=var_1, verbose=var_1, return_output=var_1, ignore_errors=var_1)

def test_case_6():
    var_0 = 'ls'
    var_1 = [var_0, var_0]
    module_0.run_command(var_1)

def test_case_7():
    var_0 = 'echo'
    var_1 = module_0.run_command(var_0, verbose=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_2 = module_0.run_command(var_0, return_output=var_1, ignore_errors=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3

def test_case_8():
    var_0 = 'ls'
    var_1 = module_0.run_command(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_9():
    var_0 = 'echo'
    var_1 = [var_0, var_0]
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
    var_4 = None
    var_5 = False
    var_6 = module_0.run_command(var_0, timeout=var_2, verbose=var_4, ignore_errors=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    var_7 = 'ls'
    var_8 = module_0.run_command(var_1, verbose=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = '/n2nexissent'
    var_10 = [var_7, var_9, var_9]
    var_11 = [var_4, var_4, var_4]
    var_12 = module_0.CommandResult(*var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_12) == 3
    var_13 = module_0.run_command(var_10, ignore_errors=var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_13) == 3

def test_case_10():
    var_0 = 'echo'
    var_1 = [var_0, var_0]
    var_2 = module_0.run_command(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_3 = [var_0, var_0]
    var_4 = True
    var_5 = module_0.run_command(var_3, return_output=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_5) == 3
    var_6 = None
    var_7 = module_0.error_wrapper(var_4)
    assert var_7 is True
    var_8 = False
    var_9 = module_0.run_command(var_0, timeout=var_4, verbose=var_6, ignore_errors=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_9) == 3
    var_10 = module_0.run_command(var_3, verbose=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3
    var_11 = True
    var_12 = module_0.run_command(var_3, env=var_6, verbose=var_11, return_output=var_6, ignore_errors=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_12) == 3
    var_13 = module_0.run_command(var_1, timeout=var_8, ignore_errors=var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_13) == 3