# Check out: https://github.com/GlowCheese/deepmosa
import flutes.fs as module_0
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = -2122
    module_0.readable_size(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.readable_size(var_0)

def test_case_3():
    var_0 = "la+3\rsB4zs'Al\x0cW"
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '$q_c[5u4oiCrh:)r.'
    var_1 = module_0.remove_suffix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = False
    module_0.copy_tree(var_2, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = 'dbMcJI:x5~'
    var_2 = module_0.remove_suffix(var_1, var_1, var_0)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    module_0.copy_tree(var_0, var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.cache(var_0, var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_0.get_folder_size(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.get_folder_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.get_file_lines(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ']l7cJYm6f$@H.'
    var_1 = True
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '1.00'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.cache(var_3)
    var_5 = module_0.remove_prefix(var_0, var_0)
    assert var_5 == ''
    var_6 = -876.371
    var_7 = module_0.readable_size(var_6)
    assert var_7 == '-876.37'
    var_8 = 'J7BO4iQ@miD"'
    var_9 = module_0.remove_prefix(var_7, var_8)
    assert var_9 == '-876.37'
    var_10 = None
    var_11 = module_0.scandir(var_10)
    module_0.readable_size(var_4)

def test_case_10():
    var_0 = 1071
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.05K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = ']l7cJYm6f$@H.'
    var_3 = False
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '0.00'
    var_5 = module_0.remove_prefix(var_2, var_2)
    assert var_5 == ''

def test_case_11():
    var_0 = ']l7cJYm6f$@H.'
    var_1 = False
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '0.00'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = ' &}"p8B'
    var_4 = False
    var_5 = module_0.remove_prefix(var_3, var_0, var_4)
    assert var_5 == ' &}"p8B'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = 'dbMcJI:x5~'
    var_2 = module_0.remove_suffix(var_1, var_1, var_0)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = 924.5
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '924.50'
    var_5 = 'f!dSl7\x0cBk;(eAy)P,'
    var_6 = module_0.remove_prefix(var_2, var_5)
    assert var_6 == ''
    var_7 = module_0.scandir(var_0)
    var_8 = module_0.scandir(var_0)
    var_9 = False
    var_10 = module_0.remove_suffix(var_6, var_5, var_9)
    assert var_10 == ''
    var_11 = module_0.readable_size(var_3)
    assert var_11 == '924.50'
    var_12 = "'YQB~T#$2<3{^w<_"
    var_13 = module_0.remove_prefix(var_12, var_11)
    assert var_13 == "'YQB~T#$2<3{^w<_"
    var_14 = None
    var_15 = module_0.scandir(var_14)
    var_16 = 99
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '99.00'
    var_18 = module_0.cache(var_14, name=var_14)
    var_19 = module_0.scandir(var_14)
    module_0.readable_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = 'dbMcJI:x5~'
    var_2 = module_0.remove_suffix(var_1, var_1, var_0)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = ']o?gU&O4k&B`<\x0c'
    var_4 = 'KJ@\\28i8[/\\'
    var_5 = False
    var_6 = module_0.remove_suffix(var_3, var_4, var_5)
    assert var_6 == ']o?gU&O4k&B`<\x0c'
    var_7 = None
    var_8 = module_0.scandir(var_7)
    module_0.readable_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = True
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.remove_prefix(var_1, var_1)
    assert var_2 == ''
    var_3 = None
    var_4 = module_0.scandir(var_3)
    var_5 = module_0.scandir(var_3)
    var_6 = '\\NwO:1='
    var_7 = True
    var_8 = module_0.remove_suffix(var_1, var_6, var_7)
    assert var_8 == '1.00'
    module_0.readable_size(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = ']l7cJYm6f$@H.'
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = '\nt__'
    var_3 = module_0.remove_prefix(var_2, var_0)
    assert var_3 == '\nt__'
    var_4 = None
    var_5 = module_0.scandir(var_4)
    var_6 = 'U4>hrT(b]6}\\xS'
    var_7 = module_0.cache(var_4, name=var_6)
    module_0.get_file_lines(var_4)