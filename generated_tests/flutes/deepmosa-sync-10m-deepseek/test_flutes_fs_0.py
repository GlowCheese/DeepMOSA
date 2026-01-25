# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.fs as module_0
import platform as module_1

def test_case_0():
    pass

def test_case_1():
    var_0 = -1026.101
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '-1026.10'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "la+3\rsB4zs'Al\x0cW"
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = module_0.scandir(var_2)
    module_0.get_folder_size(var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '$q_c[5u4oiCrh:)r.'
    var_1 = module_0.remove_suffix(var_0, var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = False
    module_0.copy_tree(var_2, var_2, var_3)

def test_case_4():
    var_0 = '/some/string/path'
    var_1 = '[H{'
    var_2 = None
    var_3 = module_0.remove_suffix(var_0, var_1, var_2)
    assert var_3 == '/some/string/path'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.scandir(var_0)
    with pytest.raises(FileNotFoundError):
        var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.cache(var_0, var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_0.get_folder_size(var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.get_folder_size(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.get_file_lines(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = "la+3\rsB4zs'Al\x0cW"
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = module_0.remove_suffix(var_1, var_0)
    assert var_3 == ''
    module_0.readable_size(var_2)

def test_case_9():
    var_0 = module_1.python_revision()
    assert var_0 == ''
    var_1 = '!^UMx'
    var_2 = '3'
    var_3 = module_0.remove_prefix(var_1, var_2)
    assert var_3 == '!^UMx'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = True
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00'

def test_case_10():
    var_0 = '/tmp'
    var_1 = module_0.cache(var_0, name=var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = module_1.uname()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'platform.uname_result'
    assert len(var_2) == 6
    assert f'{type(module_1.uname_result.processor).__module__}.{type(module_1.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_1.uname_result.processor.attrname == 'processor'
    assert f'{type(module_1.uname_result.processor.lock).__module__}.{type(module_1.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    var_3 = module_0.scandir(var_0)
    var_4 = next(var_3)
    var_5 = module_0.copy_tree(var_4, var_4, var_4)

def test_case_11():
    var_0 = "la+3\rsB4zs'Al\x0cW"
    var_1 = module_0.cache(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = True
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00'
    var_4 = module_0.remove_prefix(var_0, var_0)
    assert var_4 == ''
    var_5 = None
    var_6 = 851
    var_7 = module_0.remove_suffix(var_4, var_0)
    assert var_7 == ''
    var_8 = module_0.scandir(var_5)
    var_9 = module_0.readable_size(var_6)
    assert var_9 == '851.00'
    var_10 = 2417.76743
    var_11 = module_0.readable_size(var_10)
    assert var_11 == '2.36K'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = "la+3\rsB4zs'Al\x0cW"
    var_1 = module_0.remove_prefix(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = module_1.python_implementation()
    assert var_3 == 'CPython'
    var_4 = var_3.__repr__()
    assert var_4 == "'CPython'"
    var_5 = var_3.swapcase()
    assert var_5 == 'cpYTHON'
    var_6 = var_3.__le__(var_2)
    var_7 = True
    var_8 = module_0.cache(var_4, var_7)
    var_9 = '{j%GF_=zfV-q`\x0b'
    var_10 = 'D+'
    var_11 = module_0.remove_prefix(var_9, var_10, var_1)
    assert var_11 == '{j%GF_=zfV-q`\x0b'
    var_12 = '"x4D2\x0cLk\t\\_'
    var_13 = module_0.remove_prefix(var_4, var_12)
    assert var_13 == "'CPython'"
    module_0.copy_tree(var_3, var_5, var_5)

def test_case_13():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00P'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_6 = var_0 * var_0
    var_7 = var_6 * var_0
    var_8 = var_1 * var_0
    var_9 = 10.125
    var_10 = var_8 * var_9
    var_11 = module_0.readable_size(var_10)
    assert var_11 == '10.12G'

def test_case_14():
    var_0 = '/some/string/path'
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(FileNotFoundError):
        var_2 = list(var_1)

def test_case_15():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '/tmp'
    module_0.copy_tree(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = next(var_1)
    module_0.copy_tree(var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.scandir(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = next(var_1)
    module_0.copy_tree(var_2, var_2)