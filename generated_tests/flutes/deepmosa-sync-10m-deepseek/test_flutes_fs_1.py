# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.fs as module_0
import platform as module_1

def test_case_0():
    pass

def test_case_1():
    var_0 = 2095
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '2.05K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_2():
    var_0 = 'V&g /?_HP_(E\rY*e'
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = 1899.0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.85K'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "'u\x0bi=74\\a'\x0b\x0b/2U"
    var_1 = None
    var_2 = module_0.remove_prefix(var_0, var_0, var_1)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.get_folder_size(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = ''
    var_2 = module_0.remove_suffix(var_1, var_1)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.readable_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_1.python_implementation()
    assert var_1 == 'CPython'
    var_2 = module_0.remove_suffix(var_1, var_1, var_0)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.readable_size(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '/-F9,|'
    var_1 = module_0.remove_suffix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = -99
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '-99.00'
    var_4 = None
    var_5 = module_0.scandir(var_4)
    module_0.get_file_lines(var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '9.h'
    var_1 = module_0.cache(var_0, name=var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = True
    module_0.copy_tree(var_2, var_0, var_3)

def test_case_8():
    var_0 = 'non_existent_directory'
    var_1 = None
    var_2 = module_0.cache(var_1)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.scandir(var_0)
    with pytest.raises(FileNotFoundError):
        var_4 = list(var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.get_folder_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.get_file_lines(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 1607
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.57K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = '^1Nb7`'
    var_3 = module_0.remove_suffix(var_2, var_1)
    assert var_3 == '^1Nb7`'
    var_4 = 'p@-QSei"\tD.fnzwyux'
    var_5 = module_0.remove_suffix(var_1, var_4)
    assert var_5 == '1.57K'
    module_0.get_file_lines(var_5)

def test_case_12():
    var_0 = '/tmp'
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = 'x'
    var_3 = module_0.remove_prefix(var_0, var_2)
    assert var_3 == '/tmp'
    var_4 = None
    var_5 = var_1.__gt__(var_4)
    var_6 = list(var_1)

def test_case_13():
    var_0 = 'non_existent_directory'
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(FileNotFoundError):
        var_2 = list(var_1)

def test_case_14():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00P'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_15():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '/tmp'
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_0.copy_tree(var_2, var_0)