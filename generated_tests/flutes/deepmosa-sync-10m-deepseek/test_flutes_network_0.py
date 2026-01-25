# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

def test_case_1():
    var_0 = 'test.txt'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/test.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    module_0.download(var_0, var_0, extract=var_0, **var_0)

def test_case_3():
    var_0 = 'https://drive.google.com/file/d/ghi789'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, extract=var_1)
    assert var_2 == '/tmp/ghi789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.download(var_2, filename=var_2, extract=var_2)
    assert var_3 == '/tmp/ghi789'

def test_case_4():
    var_0 = 'http://example.com/file.txt'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/file.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'custom.bin'
    var_1 = None
    module_0.download(var_0, extract=var_1, progress=var_0, bar_fn=var_1)

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = None
    var_2 = module_0.download(var_0, extract=var_1)
    assert var_2 == '/tmp/abc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/file/d/complex_id_123_ab/details'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/complex_id_123_ab'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = '/tmp'
    module_0._download_from_google_drive(var_0, var_2, var_2, var_2)
    assert var_3 == '/tmp/test.txt'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'test.txt'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/test.txt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = '/tmp'
    module_0._download_from_google_drive(var_1, var_0, var_2, var_1)
    assert var_3 == '/tmp/test.txt'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https://drive.google.com/file/d/complex_id_123_abc/details'
    var_1 = 'test.txt'
    var_2 = None
    var_3 = 'hcA7O\x0ca'
    var_4 = '1>'
    var_5 = 'eo>T_A@W,W8(E9Z#oK\x0b'
    var_6 = {var_3: var_1, var_1: var_0, var_4: var_0, var_5: var_3}
    var_7 = module_0.download(var_0, filename=var_2, extract=var_1, progress=var_1, **var_6)
    assert var_7 == '/tmp/complex_id_123_abc'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_8 = module_0.download(var_1, var_2, bar_fn=var_2)
    assert var_8 == '/tmp/test.txt'
    module_0.download(var_8, extract=var_8, progress=var_8, bar_fn=var_6)