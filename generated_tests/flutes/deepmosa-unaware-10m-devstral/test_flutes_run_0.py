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
    var_0 = 'sleep'
    var_1 = 'et9~\\6K_;J Z-T+z'
    var_2 = [var_0, var_1]
    var_3 = 0.01
    module_0.run_command(var_2, timeout=var_3)

def test_case_6():
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
    var_4 = module_0.run_command(var_1, verbose=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3

def test_case_7():
    var_0 = 'ls'
    var_1 = module_0.run_command(var_0, ignore_errors=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_1) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_8():
    var_0 = True
    var_1 = 'ls'
    var_2 = [var_1, var_1]
    var_3 = module_0.run_command(var_2, ignore_errors=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_9():
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
def test_case_10():
    var_0 = 'echo'
    var_1 = 'test'
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
    var_6 = False
    var_7 = module_0.run_command(var_5, return_output=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3
    var_8 = 'false'
    var_9 = [var_8]
    var_10 = module_0.run_command(var_9, ignore_errors=var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3
    var_11 = 'sleep'
    var_12 = '10'
    var_13 = [var_11, var_12]
    var_14 = 0.1
    var_15 = module_0.run_command(var_13, timeout=var_14, ignore_errors=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_15) == 3
    var_16 = 'env'
    var_17 = [var_16]
    var_18 = 'TEST_VAR'
    var_19 = 'test_value'
    var_20 = {var_18: var_19}
    var_21 = module_0.run_command(var_17, env=var_20, return_output=var_3)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_21) == 3
    var_22 = 'pwd'
    var_23 = 'utf-8'
    var_24 = 'verboseTtest'
    var_25 = [var_22, var_24]
    var_26 = module_0.run_command(var_25, verbose=var_23, return_output=var_23)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_26) == 3
    var_27 = 'echo string_test'
    module_0.run_command(var_27, return_output=var_23)