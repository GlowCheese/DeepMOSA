# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

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
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    var_1 = '/tmp/test_download'
    var_2 = 'test_file.txt'
    var_3 = module_0.download(var_0, var_1, var_2)
    assert var_3 == '/tmp/test_download/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_1.exists(var_3)
    assert var_4 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_5 = module_2.basename(var_3)
    assert var_5 == 'test_file.txt'
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'https://drive.google.com/file/d/1A2B3CSD5E6FG8H9I0J/view'
    module_0.download(var_0, filename=var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download_from_google_drive(var_0, var_1, var_2)
    assert var_3 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.download(var_3, filename=var_3)
    assert var_4 == '/tmp/test_file.txt'
    var_5 = module_0.download(var_3, extract=var_3, progress=var_4)
    assert var_5 == '/tmp/test_file.txt'
    module_0.download(var_3, extract=var_2, progress=var_5, bar_fn=var_5)

def test_case_6():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'https://drive.google.com/file/d/1A2B3CSD5E6FG8H9I0J/view'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, bar_fn=var_1)
    assert var_2 == '/tmp/1A2B3CSD5E6FG8H9I0J'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_1.exists(var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6FG8H90J/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download_from_google_drive(var_0, var_1, var_2)
    assert var_3 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.download(var_3, extract=var_3, progress=var_1)
    assert var_4 == '/tmp/test_file.txt'
    var_5 = module_1.exists(var_4)
    assert var_5 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_6 = module_0.download(var_0, extract=var_4)
    assert var_6 == '/tmp/1A2B3C4D5E6FG8H90J'
    var_7 = None
    module_0.download(var_3, var_7, var_7, progress=var_7, **var_6)