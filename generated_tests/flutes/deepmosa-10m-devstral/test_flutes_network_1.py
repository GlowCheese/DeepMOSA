# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1
import urllib.request as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'W":z}F Pc!K5)H'
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1}
    module_0.download(var_0, var_1, progress=var_0, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ' 5r4n^]%L O9LR_&.1'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

def test_case_3():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.download(var_3, var_2, var_4, bar_fn=var_3)
    assert var_5 == '/tmp/test_file.txt'
    var_6 = var_4.__str__()
    assert var_6 == '/tmp/test_file.txt'
    var_7 = module_1.exists(var_4)
    assert var_7 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)

def test_case_5():
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_2.HTTPPasswordMgrWithDefaultRealm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'urllib.request.HTTPPasswordMgrWithDefaultRealm'
    assert var_0.passwd == {}
    assert module_2.MAXFTPCACHE == 10
    assert module_2.ftpcache == {}
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_7():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharing'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/123456789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_0._download_from_google_drive(var_0, var_0, var_1, var_2)
    assert var_3 == '/tmp/test_file.txt'

def test_case_10():
    var_0 = 'https://eample.com/fil.txt'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)
    assert var_2 == '/tmp/fil.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.split()

def test_case_11():
    var_0 = 'https://drive.google.com/file/d/245678/view?usp=sharing'
    var_1 = None
    var_2 = 'o'
    var_3 = {var_2: var_1, var_2: var_1, var_0: var_1, var_2: var_1}
    var_4 = module_0.download(var_0, filename=var_1, bar_fn=var_1, **var_3)
    assert var_4 == '/tmp/245678'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_5 = 'test_file.txt'
    var_6 = '/tmp'
    var_7 = module_0._download_from_google_drive(var_0, var_5, var_6, var_1)
    assert var_7 == '/tmp/test_file.txt'
    var_8 = bool(var_4)
    assert var_8 is True

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharng'
    var_1 = 'testD_file.tx'
    module_0._download_from_google_drive(var_0, var_1, var_1, var_0)
    assert var_2 == '/tmp/test_file.txt'