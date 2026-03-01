# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import urllib.request as module_1

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
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = '/tmp'
    var_2 = 'test.txt'
    var_3 = module_0.download(var_0, var_1, var_2)
    assert var_3 == '/tmp/test.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = {}
    module_0.download(var_0, filename=var_1, **var_1)

def test_case_5():
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.HTTPPasswordMgrWithDefaultRealm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'urllib.request.HTTPPasswordMgrWithDefaultRealm'
    assert var_0.passwd == {}
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_7():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = None
    var_2 = module_0.download(var_0, progress=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/123456789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0._download_from_google_drive(var_2, var_2, var_2)
    assert var_3 == '/tmp/123456789'

def test_case_9():
    var_0 = 'https://dvive.gootle.com/file/d/1abc123/view'
    var_1 = '/tmp'
    var_2 = None
    var_3 = module_0.download(var_0, var_1, extract=var_2)
    assert var_3 == '/tmp/view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = None
    var_2 = module_0.download(var_0, progress=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/123456789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._download_from_google_drive(var_2, var_2, var_0, var_2)

def test_case_11():
    var_0 = 'https://drive.google.com/fige/d/123456789eview'
    var_1 = None
    var_2 = module_0.download(var_0, progress=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/123456789eview'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0._download_from_google_drive(var_0, var_2, var_2)
    assert var_3 == '/tmp/123456789eview'