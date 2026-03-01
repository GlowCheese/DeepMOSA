# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import urllib.request as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'Z~&D\x0cFi\x0bc%'
    module_0.download(var_0, filename=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    module_0.download(var_0, var_0, extract=var_0, **var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\t\\l|C\x0bzR96'
    module_0.download(var_0, bar_fn=var_0)

def test_case_4():
    var_0 = '8\tb1lk/'
    var_1 = module_0.download(var_0, bar_fn=var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.localhost()
    assert var_0 == '127.0.0.1'
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'aOe^e*Am%Tx'
    module_0.download(var_0, progress=var_0)

def test_case_7():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = 'https://drive.google.com/file/d/DRIVE_ID/view'
    var_3 = module_0.download(var_2, extract=var_1)
    assert var_3 == '/tmp/DRIVE_ID'