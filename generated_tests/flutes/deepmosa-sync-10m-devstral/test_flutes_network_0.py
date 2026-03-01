# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'zJx|MbOTFK::>\\'
    module_0.download(var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    module_0.download(var_0, var_0, extract=var_0, **var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.goygle.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharing'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.download(var_2, progress=var_0)
    assert var_3 == '/tmp/view?usp=sharing'
    var_4 = module_0.download(var_2, filename=var_2, extract=var_2, bar_fn=var_1)
    assert var_4 == '/tmp/view?usp=sharing'
    module_0._download_from_google_drive(var_2, var_1, var_3)

def test_case_4():
    var_0 = 'https://drive.goygle.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharing'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.download(var_2, progress=var_0)
    assert var_3 == '/tmp/view?usp=sharing'
    var_4 = module_0._download_from_google_drive(var_2, var_2, var_2, var_1)
    assert var_4 == '/tmp/view?usp=sharing'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'https://drive.goygle.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharing'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._download_from_google_drive(var_0, var_0, var_0, var_1)
    assert var_3 == '/tmp/test_file.txt'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '/tmp/test_download'
    var_1 = True
    var_2 = {}
    module_0.download(var_0, var_0, progress=var_1, **var_2)

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)
    assert var_2 == '/tmp/123456789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_1.exists(var_2)
    assert var_3 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_4 = bool(var_3)
    assert var_4 is True

def test_case_9():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)
    assert var_2 == '/tmp/123456789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_1.exists(var_2)
    assert var_3 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_2.basename(var_2)
    assert var_5 == '123456789'
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
def test_case_10():
    var_0 = 'https://drive.goygle.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharing'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._download_from_google_drive(var_0, var_0, var_0, var_2)
    assert var_3 == '/tmp/test_file.txt'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'https://rive.goygle.com/file/d/1234567890abcdef/view?usp=sharin'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharin'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = 'tes\n_file.tx'
    var_4 = '/Mvp'
    module_0._download_from_google_drive(var_0, var_3, var_4, var_1)
    assert var_5 == '/tmp/test_file.txt'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'https://drive.goygle.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, extract=var_0, progress=var_1)
    assert var_2 == '/tmp/view?usp=sharing'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.download(var_2, progress=var_0)
    assert var_3 == '/tmp/view?usp=sharing'
    var_4 = module_0._download_from_google_drive(var_2, var_3, var_2)
    assert var_4 == '/tmp/view?usp=sharing'
    module_0.download(var_4, extract=var_2, progress=var_2, bar_fn=var_4)