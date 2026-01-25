# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.fs as module_0

def test_case_0():
    pass

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
    var_3 = None
    module_0.get_folder_size(var_3)

def test_case_4():
    var_0 = 'om_ZTNUP6\nI~|1e'
    var_1 = module_0.remove_suffix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.cache(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = module_0.scandir(var_2)
    var_4 = 'om {-ab7dj'
    var_5 = 'b \x0crx>onNck3'
    var_6 = module_0.remove_suffix(var_4, var_5)
    assert var_6 == 'om {-ab7dj'
    var_7 = None
    module_0.remove_suffix(var_4, var_7)

def test_case_6():
    var_0 = None
    var_1 = 'j((3oG<=ABa'
    var_2 = module_0.remove_suffix(var_1, var_1, var_0)
    assert var_2 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = ']1[p8zB|#i8\n;_eDc'
    var_4 = module_0.remove_prefix(var_3, var_3)
    assert var_4 == ''

def test_case_7():
    var_0 = 'Jej\r*~lHe"'
    var_1 = module_0.cache(var_0, name=var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = None
    var_1 = module_0.cache(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.get_folder_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.get_file_lines(var_0)

def test_case_11():
    var_0 = ''
    var_1 = module_0.remove_suffix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_12():
    var_0 = 978.2
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '978.20'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = '9+=\tP324~|16YN5.'
    var_3 = '%^%%3hZ'
    var_4 = module_0.remove_prefix(var_2, var_3)
    assert var_4 == '9+=\tP324~|16YN5.'

def test_case_13():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = var_0 * var_0
    var_3 = var_0 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00G'
    var_6 = var_0 * var_0
    var_7 = var_6 * var_0
    var_8 = var_7 * var_0
    var_9 = module_0.readable_size(var_8)
    assert var_9 == '1.00T'
    var_10 = var_0 * var_0
    var_11 = var_10 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = module_0.readable_size(var_13)
    assert var_14 == '1.00P'
    var_15 = 512
    var_16 = module_0.readable_size(var_15)
    assert var_16 == '512.00'
    var_17 = 1536
    var_18 = module_0.readable_size(var_17)
    assert var_18 == '1.50K'