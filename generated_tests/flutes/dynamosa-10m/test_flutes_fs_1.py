# Check out: https://github.com/GlowCheese/deepmosa
import platform as module_1

import flutes.fs as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.get_folder_size(var_0)

def test_case_1():
    var_0 = 1607
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.57K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_2():
    var_0 = 'V&g /?_HP_(E\rY*e'
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_3():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = module_0.remove_prefix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_4():
    var_0 = module_1.python_implementation()
    assert var_0 == 'CPython'
    var_1 = module_0.remove_suffix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_5():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = module_0.remove_suffix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 1618
    module_0.cache(var_0, name=var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.cache(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.get_file_lines(var_0)

def test_case_9():
    var_0 = 'u-"%winO#[t{$egl/ID'
    var_1 = 'R\n ZG[Pk\nTIi\\,%_G0W'
    var_2 = module_0.remove_suffix(var_0, var_1)
    assert var_2 == 'u-"%winO#[t{$egl/ID'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_10():
    var_0 = ''
    var_1 = module_0.remove_suffix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = '?B.$we[Uu#k(I2\\89'
    var_1 = '?piEU:=K*\x0bS=3)'
    var_2 = module_0.remove_prefix(var_0, var_1)
    assert var_2 == '?B.$we[Uu#k(I2\\89'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'