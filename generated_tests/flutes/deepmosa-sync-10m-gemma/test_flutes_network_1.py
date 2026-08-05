# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import urllib.request as module_0
import flutes.network as module_1
import posixpath as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.thishost()
    assert module_0.MAXFTPCACHE == 10
    assert module_0.ftpcache == {}
    module_1.download(var_0, filename=var_0, extract=var_0, progress=var_0, bar_fn=var_0)

def test_case_1():
    var_0 = 'https://drive.google.com/file/d/abc123xyz/view'
    var_1 = {}
    var_2 = module_1.download(var_0, var_0, **var_1)
    assert var_2 == 'https://drive.google.com/file/d/abc123xyz/view/abc123xyz'
    assert f'{type(module_1.PathType).__module__}.{type(module_1.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.download(var_0, var_0, var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.google.com/file/d/abc123xyz/view'
    var_1 = {}
    var_2 = module_1.download(var_0, var_0, **var_1)
    assert var_2 == 'https://drive.google.com/file/d/abc123xyz/view/abc123xyz'
    assert f'{type(module_1.PathType).__module__}.{type(module_1.PathType).__qualname__}' == 'typing.TypeVar'
    module_1.download(var_2, filename=var_2, bar_fn=var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    module_1.download(var_0, var_0, **var_0)
    assert var_1 == '/tmp/abc123xyz'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/tmp/test'
    var_1 = {}
    module_1.download(var_0, var_0, **var_1)
    assert var_2 == '/tmp/test/data.txt'

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/abc123xyz/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_1.download(var_0, var_1, **var_2)
    assert var_3 == '/tmp/test/abc123xyz'
    assert f'{type(module_1.PathType).__module__}.{type(module_1.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/test_id/view'
    var_1 = 'test_file.txt'
    var_2 = '.'
    var_3 = None
    var_4 = module_1._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == './test_file.txt'
    assert f'{type(module_1.PathType).__module__}.{type(module_1.PathType).__qualname__}' == 'typing.TypeVar'
    var_5 = [var_1]
    var_6 = module_2.join(var_2, *var_5)
    assert var_6 == './test_file.txt'
    assert module_2.curdir == '.'
    assert module_2.pardir == '..'
    assert module_2.extsep == '.'
    assert module_2.sep == '/'
    assert module_2.pathsep == ':'
    assert module_2.defpath == '/bin:/usr/bin'
    assert module_2.altsep is None
    assert module_2.devnull == '/dev/null'
    assert f'{type(module_2.ALLOW_MISSING).__module__}.{type(module_2.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_2.supports_unicode_filenames is False
    var_7 = bool(var_4 == var_6)
    assert var_7 is True
    var_8 = module_1.download(var_6, extract=var_7, progress=var_4)
    assert var_8 == '/tmp/test_file.txt'