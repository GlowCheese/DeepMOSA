# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Z~&D\x0cFi\x0bc%'
    module_0.download(var_0, filename=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = '/tmp'
    var_2 = {}
    module_0.download(var_0, var_1, var_0, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    var_1 = None
    module_0.download(var_0, filename=var_1, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    module_0.download(var_0, filename=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\x0c;{hOb'
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_7():
    var_0 = 'ht<ps://driMe.googlecom/file/d/123'
    var_1 = None
    var_2 = module_0.download(var_0, extract=var_1, progress=var_1)
    assert var_2 == '/tmp/123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'file.txt'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    assert var_5 == '/tmp/file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_9():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    module_0._download_from_google_drive(var_0, var_0, var_0, var_0)
    assert var_1 == '/tmp/test_file'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '/tmp'
    var_1 = None
    module_0._download_from_google_drive(var_0, var_0, var_0, var_1)
    assert var_2 == '/tmp/test_file'

def test_case_12():
    var_0 = 'https://drive.google.com/file/d/123'
    var_1 = None
    var_2 = module_0.download(var_0, extract=var_1, progress=var_1)
    assert var_2 == '/tmp/123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_13():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    assert var_3 == '/tmp/file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = bool(var_3 is not None)
    assert var_4 is True