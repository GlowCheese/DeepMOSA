# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import urllib.request as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'download_warning'
    module_0.download(var_0, var_0, progress=var_0)

def test_case_1():
    var_0 = 'https://drive.google.com/file/d/1abcde12345/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/1abcde12345'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, var_0, var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.google.com/d/xyz123'
    module_0.download(var_0, filename=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1.thishost()
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, filename=var_0, extract=var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'token_test.txt'
    module_0.download(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'X\n'
    module_0.download(var_0)

def test_case_7():
    var_0 = 'http://drive.google.com/d/xz13'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/xz13'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/test_id456/view'
    var_1 = module_0.download(var_0, var_0, progress=var_0)
    assert var_1 == 'https://drive.google.com/file/d/test_id456/view/test_id456'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_9():
    var_0 = 'http://drive.gooe.com/ile/d/1abcde12345/view'
    var_1 = None
    var_2 = '7]5"N^9b%@tnyfXZv'
    var_3 = {var_2: var_1, var_2: var_1, var_2: var_1}
    var_4 = module_0.download(var_0, filename=var_1, progress=var_1, **var_3)
    assert var_4 == '/tmp/view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https/drive.google.com/d/yz123'
    var_1 = module_0.download(var_0, var_0, extract=var_0)
    assert var_1 == 'https/drive.google.com/d/yz123/yz123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.download(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'https://drive.google.com/file/d/test_id456-view'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, var_1, var_1, var_1, var_1)
    assert var_2 == '/tmp/test_id456-view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.download(var_1, filename=var_2)
    assert var_3 == '/tmp/test_id456-view'
    var_4 = module_1.noheaders()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'email.message.Message'
    assert len(var_4) == 0
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_1, extract=var_1, bar_fn=var_4, **var_1)