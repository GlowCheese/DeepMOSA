# Check out: https://github.com/GlowCheese/deepmosa
import flutes.run as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.error_wrapper(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.run_command(var_0, timeout=var_0, verbose=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '9osC'
    module_0.run_command(var_0, env=var_0, cwd=var_0, verbose=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = True
    var_2 = ''
    var_3 = '+.9n>PHMD'
    var_4 = {var_2: var_0, var_3: var_0, var_2: var_0}
    module_0.run_command(var_0, env=var_0, verbose=var_1, ignore_errors=var_0, **var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.CommandResult()

def test_case_5():
    var_0 = 'Hello, World!'
    var_1 = 'ls'
    var_2 = 'neisen<\r_rile.tU'
    var_3 = 'slee|'
    var_4 = 'M0'
    var_5 = [var_1, var_0, var_1, var_3, var_3, var_0, var_2, var_4]
    var_6 = 1
    module_0.run_command(var_5, timeout=var_6)

def test_case_6():
    var_0 = 'echo'
    var_1 = 'Hello, World!'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.run_command(var_2, return_output=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = 'ls'
    var_6 = 'neisent\r_file.tx'
    var_7 = [var_5, var_6]
    var_8 = module_0.run_command(var_7, ignore_errors=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = None
    var_10 = module_0.run_command(var_5, cwd=var_9, verbose=var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3
    module_0.run_command(var_7, timeout=var_3)

def test_case_7():
    var_0 = True
    var_1 = 'ls'
    var_2 = None
    var_3 = module_0.error_wrapper(var_2)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    var_4 = module_0.run_command(var_1, cwd=var_3, verbose=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = module_0.run_command(var_1, timeout=var_0, verbose=var_3, return_output=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_5) == 3

def test_case_8():
    var_0 = 'echo'
    var_1 = 'Hello, World!'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.run_command(var_2, return_output=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = 'ls'
    var_6 = '/nonexistent'
    var_7 = [var_5, var_6]
    var_8 = module_0.run_command(var_7, ignore_errors=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = 'sleep'
    var_10 = '10'
    var_11 = [var_9, var_10]
    var_12 = 0.1
    var_13 = module_0.run_command(var_11, timeout=var_12, ignore_errors=var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_13) == 3
    var_14 = 'printenv'
    var_15 = 'MY_VAR'
    var_16 = [var_14, var_15]
    var_17 = 'test_value'
    var_18 = {var_15: var_17}
    var_19 = module_0.run_command(var_16, env=var_18, return_output=var_3)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_19) == 3
    var_20 = 'All tests passed!'
    var_21 = print(var_20)

def test_case_9():
    var_0 = 'ls'
    var_1 = None
    var_2 = module_0.run_command(var_0, cwd=var_1, verbose=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'echo'
    var_1 = 'Hello, World!'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.run_command(var_2, return_output=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = 'false'
    var_6 = [var_5]
    var_7 = module_0.run_command(var_6, ignore_errors=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3
    var_8 = 'sleep'
    var_9 = '2'
    var_10 = [var_8, var_9]
    var_11 = module_0.run_command(var_10, timeout=var_3, ignore_errors=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_11) == 3
    var_12 = 'printenv'
    var_13 = 'MY_VAR'
    var_14 = [var_12, var_13]
    var_15 = 'test_value'
    var_16 = {var_13: var_15}
    var_17 = module_0.run_command(var_14, env=var_16, return_output=var_3)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_17) == 3
    var_18 = 'yes'
    var_19 = 'A'
    var_20 = 100
    var_21 = var_19 * var_20
    var_22 = [var_18, var_21]
    var_23 = 0.1
    var_24 = module_0.run_command(var_22, timeout=var_23, ignore_errors=var_3)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_24) == 3
    var_25 = 'echo Hello, World!'
    module_0.run_command(var_25, return_output=var_3)