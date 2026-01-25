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
    var_0 = ''
    module_0.run_command(var_0, env=var_0, verbose=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'j+]'
    module_0.run_command(var_0, cwd=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\r4j\nO7%b*b\rutn:^A4,'
    module_0.run_command(var_0, env=var_0, verbose=var_0)

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
    var_0 = 'false'
    var_1 = [var_0]
    var_2 = {}
    module_0.run_command(var_1, **var_2)

def test_case_7():
    var_0 = 'echo hello'
    var_1 = True
    var_2 = 'shell'
    var_3 = {var_2: var_1}
    var_4 = module_0.run_command(var_0, return_output=var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = bool(b'hello' in var_4.captured_output)
    assert var_5 is True

def test_case_8():
    var_0 = 'echo'
    var_1 = None
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
    var_3 = {}
    var_4 = module_0.run_command(var_2, verbose=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = var_4.captured_output

def test_case_10():
    var_0 = 'false'
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = module_0.run_command(var_1, ignore_errors=var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.run.CommandResult'
    assert len(var_4) == 3
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    assert f'{type(module_0.CommandResult.command).__module__}.{type(module_0.CommandResult.command).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.return_code).__module__}.{type(module_0.CommandResult.return_code).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.CommandResult.captured_output).__module__}.{type(module_0.CommandResult.captured_output).__qualname__}' == '_collections._tuplegetter'
    var_5 = bool(var_4.return_code != 0)
    assert var_5 is True
    var_6 = var_4.captured_output
    var_7 = bool(var_4.captured_output is not None)
    assert var_7 is True

def test_case_11():
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
    var_8 = var_6.captured_output
    var_9 = bool(var_6.captured_output is not None)
    assert var_9 is True

def test_case_12():
    var_0 = "3.x'^\x0c$"
    var_1 = 10000
    var_2 = var_0 * var_1
    var_3 = 'python3'
    var_4 = '-c'
    var_5 = f"print('{var_2}')"
    var_6 = [var_3, var_4, var_5]
    var_7 = True
    var_8 = {}
    module_0.run_command(var_6, return_output=var_7, **var_8)