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
    module_0.run_command(var_0, env=var_0, timeout=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '75Dn'
    module_0.run_command(var_0, cwd=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = True
    module_0.run_command(var_0, verbose=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.CommandResult()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '\x0c'
    module_0.run_command(var_0, cwd=var_0, verbose=var_0, ignore_errors=var_0)

def test_case_6():
    var_0 = None
    var_1 = 'dd'
    var_2 = module_0.error_wrapper(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ExcType).__module__}.{type(module_0.ExcType).__qualname__}' == 'typing.TypeVar'
    assert module_0.MAX_OUTPUT_LENGTH == 8192
    var_3 = 'u1rW0\t{b;'
    var_4 = [var_1, var_3]
    module_0.run_command(var_4, env=var_0, timeout=var_0)