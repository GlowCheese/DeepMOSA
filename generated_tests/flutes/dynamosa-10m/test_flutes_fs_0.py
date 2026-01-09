# Check out: https://github.com/GlowCheese/deepmosa
import platform as module_1

import flutes.fs as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_1():
    var_0 = False
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '0.00'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_2():
    var_0 = 'o>lj1)T:`57tc6z'
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_3():
    var_0 = 'Op{a'
    var_1 = module_0.remove_suffix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_4():
    var_0 = 'O0UPRq,~f#`=\r['
    var_1 = None
    var_2 = module_0.remove_suffix(var_0, var_0, var_1)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_5():
    var_0 = 'i[Oj!E#?3CCXguOUP'
    var_1 = module_0.cache(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.get_folder_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.get_file_lines(var_0)

def test_case_8():
    var_0 = '`>y*IG'
    var_1 = 'a&v}X5TzA%p\\cF'
    var_2 = module_0.remove_suffix(var_0, var_1)
    assert var_2 == '`>y*IG'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_9():
    var_0 = ''
    var_1 = module_0.remove_suffix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_10():
    var_0 = 1085
    var_1 = module_0.readable_size(var_0, var_0)
    assert var_1 == '1.05957031250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = 'o>lj1)T:`57tc6z'
    var_1 = 'MHJ"t'
    var_2 = module_0.remove_prefix(var_0, var_1)
    assert var_2 == 'o>lj1)T:`57tc6z'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_12():
    var_0 = "la+3\rsBz'Al\x0c#W"
    var_1 = False
    var_2 = module_0.remove_prefix(var_0, var_0, var_1)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_13():
    var_0 = '#\x0b~j=1N`@6BZD`C:ysG'
    var_1 = module_0.cache(var_0, name=var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_14():
    var_0 = None
    var_1 = module_1.libc_ver(version=var_0)
    var_2 = var_1.__hash__()
    assert var_2 == 3794758312440384309
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '3370.42P'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'