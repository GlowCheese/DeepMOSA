# Check out: https://github.com/GlowCheese/deepmosa
import urllib.request as module_1

import flutes.network as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0R/view'
    module_0.download(var_0, filename=var_0, extract=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ' 5r4n^]%L O9LR_&.1'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'r2.R'
    module_0.download(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1.HTTPPasswordMgrWithDefaultRealm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'urllib.request.HTTPPasswordMgrWithDefaultRealm'
    assert var_0.passwd == {}
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_5():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'W":z}F Pc!K5)H'
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1}
    module_0.download(var_0, var_1, progress=var_0, **var_2)

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0R/view'
    var_1 = 'r2.i'
    var_2 = module_0.download(var_0, filename=var_1, extract=var_1, progress=var_1)
    assert var_2 == '/tmp/r2.i'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/file_id/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    assert var_3 == '/tmp/test/file_id'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view'
    var_1 = '/tmp/test'
    var_2 = {}
    module_0.download(var_0, var_1, var_0, progress=var_2, **var_2)

def test_case_10():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view'
    var_1 = '/tmp/test'
    var_2 = "yr`b>J'C$\x0b75bW2sj3"
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, progress=var_0, **var_3)
    assert var_4 == "/tmp/test/yr`b>J'C$\x0b75bW2sj3"
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = lambda : var_3
    module_0._download(var_0, var_1, var_2, var_4)