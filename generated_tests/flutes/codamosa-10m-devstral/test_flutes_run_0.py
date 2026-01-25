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

def test_case_3():
    var_0 = 'echo'
    var_1 = module_0.run_command(var_0, return_output=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = None
    var_4 = module_0.run_command(var_0, timeout=var_3, verbose=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.CommandResult()

def test_case_5():
    var_0 = 'echo'
    var_1 = [var_0, var_0]
    var_2 = False
    var_3 = module_0.run_command(var_1, return_output=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_4 = module_0.run_command(var_1, verbose=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    var_5 = 'ls'
    var_6 = 'J#m%Hi\\Xi\ty\x0cMN6.l\x0bi'
    var_7 = [var_5, var_6, var_5, var_5, var_5, var_6, var_6, var_6]
    module_0.run_command(var_7, ignore_errors=var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = True
    var_1 = 'ls'
    var_2 = 'J#m%H6ai/\\Xy\x0cMN6.l\x0bi'
    var_3 = [var_1, var_2, var_1, var_1, var_1, var_2, var_2]
    var_4 = module_0.run_command(var_3, ignore_errors=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = None
    var_6 = None
    module_0.run_command(var_6, env=var_6, cwd=var_1, timeout=var_0, verbose=var_5, ignore_errors=var_6)

def test_case_7():
    var_0 = 'ls'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.run_command(var_0, env=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'echo'
    var_1 = 'hello'
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
    var_5 = [var_0, var_1]
    var_6 = module_0.run_command(var_5, verbose=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    var_7 = 'ls'
    var_8 = '/nonexistent'
    var_9 = [var_7, var_8]
    var_10 = module_0.run_command(var_9, ignore_errors=var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3
    var_11 = module_0.run_command(var_5, timeout=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_11) == 3
    var_12 = 'sleep'
    var_13 = '10'
    var_14 = [var_12, var_13]
    var_15 = -0.12776235653833712
    var_16 = module_0.run_command(var_14, timeout=var_15, ignore_errors=var_11)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_16) == 3
    var_17 = 'echo $TEST_VAR'
    var_18 = 'TEST_VAR'
    var_19 = 'test_value'
    var_20 = {var_18: var_19}
    module_0.run_command(var_17, env=var_20)