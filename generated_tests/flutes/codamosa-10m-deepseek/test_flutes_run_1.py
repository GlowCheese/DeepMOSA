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
    module_0.run_command(var_0, return_output=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '9osC'
    var_1 = None
    module_0.run_command(var_0, env=var_1, cwd=var_0, verbose=var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '9osC'
    var_1 = True
    module_0.run_command(var_0, env=var_1, cwd=var_0, verbose=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = True
    var_2 = ''
    var_3 = '+.9n>PHMD'
    var_4 = {var_2: var_0, var_3: var_0, var_2: var_0}
    module_0.run_command(var_0, env=var_0, verbose=var_1, ignore_errors=var_0, **var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.CommandResult()

def test_case_6():
    var_0 = '"u\x0b)qi\x0c\'mb63L\r\'EcHHB'
    var_1 = True
    var_2 = 'ls'
    var_3 = [var_2, var_2, var_2, var_0, var_0, var_0, var_0, var_0]
    var_4 = module_0.run_command(var_3, return_output=var_1, ignore_errors=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = 'pwd'
    var_6 = [var_5, var_5]
    var_7 = module_0.run_command(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'echo'
    var_1 = [var_0, var_0]
    var_2 = module_0.run_command(var_1, return_output=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_3 = None
    var_4 = module_0.error_wrapper(var_3)
    var_5 = module_0.run_command(var_1, timeout=var_4, return_output=var_4, ignore_errors=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_5) == 3
    module_0.run_command(var_2, env=var_4, return_output=var_4)

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
    var_6 = 'nonexistent_file'
    var_7 = [var_5, var_6]
    var_8 = module_0.run_command(var_7, ignore_errors=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_8) == 3
    var_9 = 'sleep'
    var_10 = '2'
    var_11 = [var_9, var_10]
    var_12 = module_0.run_command(var_11, timeout=var_3, ignore_errors=var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_12) == 3
    var_13 = 'printenv'
    var_14 = 'MY_VAR'
    var_15 = [var_13, var_14]
    var_16 = 'test_value'
    var_17 = {var_14: var_16}
    var_18 = module_0.run_command(var_15, env=var_17, return_output=var_3)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_18) == 3
    var_19 = 'All tests passed!'
    var_20 = print(var_19)

def test_case_9():
    var_0 = 'echo'
    var_1 = [var_0, var_0]
    var_2 = module_0.run_command(var_1, return_output=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_3 = None
    var_4 = module_0.error_wrapper(var_3)
    var_5 = 'sleep'
    var_6 = module_0.run_command(var_1, timeout=var_4, return_output=var_4, ignore_errors=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    var_7 = module_0.error_wrapper(var_5)
    assert var_7 == 'sleep'
    module_0.run_command(var_7, env=var_4, return_output=var_4)

def test_case_10():
    var_0 = 'ls'
    var_1 = None
    var_2 = False
    var_3 = module_0.run_command(var_0, env=var_1, timeout=var_1, return_output=var_2, ignore_errors=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_3) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_11():
    var_0 = '"u\x0b)qi\x0c-mbL\rQEcHHB'
    var_1 = True
    var_2 = 'ls'
    var_3 = [var_2, var_0, var_0, var_2, var_0, var_0]
    var_4 = module_0.run_command(var_3, return_output=var_1, ignore_errors=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = 'pwd'
    var_6 = [var_5]
    var_7 = module_0.run_command(var_6, verbose=var_1, return_output=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_7) == 3
    var_8 = 'n output captur'
    var_9 = [var_5, var_8]
    var_10 = module_0.run_command(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3

def test_case_12():
    var_0 = '"u\x0b)qi\x0c-mbLEcHHB'
    var_1 = None
    var_2 = module_0.error_wrapper(var_1)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    var_3 = module_0.error_wrapper(var_1)
    var_4 = True
    var_5 = [var_2, var_2, var_2]
    var_6 = module_0.CommandResult(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_6) == 3
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_7 = 'ls'
    var_8 = [var_7, var_0, var_0, var_7, var_0, var_0]
    var_9 = module_0.run_command(var_8, return_output=var_4, ignore_errors=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_9) == 3
    var_10 = module_0.error_wrapper(var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_10) == 3
    var_11 = 'env'
    var_12 = [var_11]
    var_13 = module_0.error_wrapper(var_2)
    var_14 = module_0.run_command(var_12, env=var_3, return_output=var_4)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_14) == 3
    var_15 = 'pwd'
    var_16 = [var_15]
    var_17 = module_0.run_command(var_16, verbose=var_4, return_output=var_4)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_17) == 3
    var_18 = 'n output captur'
    var_19 = [var_15, var_18]
    var_20 = module_0.run_command(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_20) == 3
    var_21 = module_0.CommandResult(*var_10)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_21) == 3
    var_22 = module_0.run_command(var_11, env=var_13, cwd=var_1, verbose=var_10)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_22) == 3
    var_23 = module_0.error_wrapper(var_1)