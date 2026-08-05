# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import urllib.request as module_0
import flutes.network as module_1
import urllib.response as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.thishost()
    assert module_0.MAXFTPCACHE == 10
    assert module_0.ftpcache == {}
    module_1.download(var_0, filename=var_0, extract=var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = {}
    module_1.download(var_0, var_0, var_0, **var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.download(var_0, var_0, var_0, progress=var_0, bar_fn=var_0)

def test_case_3():
    pass

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.getproxies_environment()
    assert module_0.MAXFTPCACHE == 10
    assert module_0.ftpcache == {}
    module_1.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'testfile.txt'
    var_1 = None
    module_1.download(var_0, var_1, progress=var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ''
    var_1 = module_1.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_1.PathType).__module__}.{type(module_1.PathType).__qualname__}' == 'typing.TypeVar'
    module_1.download(var_1, var_1, **var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'T]&Ol!~G[b9,5.'
    var_1 = module_2.addbase(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'urllib.response.addbase'
    assert var_1.file == 'T]&Ol!~G[b9,5.'
    assert var_1.name == '<urllib response>'
    assert var_1.delete is False
    assert var_1.fp == 'T]&Ol!~G[b9,5.'
    module_1.download(var_1, progress=var_1)