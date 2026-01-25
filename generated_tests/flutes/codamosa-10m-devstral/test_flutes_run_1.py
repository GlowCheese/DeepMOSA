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
    var_0 = '9osC'
    var_1 = None
    module_0.run_command(var_0, env=var_1, cwd=var_0, verbose=var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '9osC'
    var_1 = None
    var_2 = True
    module_0.run_command(var_0, env=var_1, cwd=var_0, verbose=var_2)

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
    var_0 = True
    var_1 = 'YIw\x0cFF p-0#4"3&]'
    var_2 = 'sleep'
    var_3 = [var_2, var_2, var_1, var_2, var_1, var_2, var_1, var_1, var_1, var_1, var_1, var_2, var_2, var_2, var_1, var_2, var_1]
    var_4 = module_0.run_command(var_3, timeout=var_0, ignore_errors=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    module_0.run_command(var_3)

def test_case_7():
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
    var_4 = module_0.run_command(var_1, ignore_errors=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3

def test_case_8():
    var_0 = 'echo'
    var_1 = True
    var_2 = module_0.run_command(var_0, ignore_errors=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_2) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'

def test_case_9():
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
    var_5 = 'sleep'
    var_6 = '10'
    var_7 = [var_5, var_6]
    var_8 = 0.1
    var_9 = module_0.run_command(var_7, timeout=var_8, ignore_errors=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_9) == 3
    var_10 = 'echo'
    var_11 = 'verbose'
    var_12 = [var_10, var_11]
    var_13 = True
    var_14 = module_0.run_command(var_12, verbose=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_14) == 3
    var_15 = False
    module_0.run_command(var_5, return_output=var_15)

def test_case_10():
    var_0 = 'echo'
    var_1 = [var_0, var_0, var_0, var_0]
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
    var_4 = module_0.run_command(var_1, ignore_errors=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    var_5 = module_0.run_command(var_1, verbose=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_5) == 3