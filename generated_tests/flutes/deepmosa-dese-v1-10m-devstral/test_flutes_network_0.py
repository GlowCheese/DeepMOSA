# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import builtins as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

def test_case_1():
    var_0 = module_1.object()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'file.txt'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.google.com/file/d/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0._download_from_google_drive(var_0, var_2, var_2)
    assert var_3 == '/tmp/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz'
    module_0.download(var_1, var_2, progress=var_2, bar_fn=var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'https://drive.gogle.c2m/file/+/1Ab2Cd3EfGhIjKlMnOpQrStUvW-Y/view?usp=sharing'
    module_0.download(var_0, filename=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'https://example.com/file.txt'
    var_1 = True
    module_0.download(var_0, progress=var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\x0c;{hOb'
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/file/d/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz/view?usp=sharing'
    var_1 = None
    module_0._download_from_google_drive(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    module_0._download_from_google_drive(var_0, var_0, var_0)

def test_case_10():
    var_0 = 'https://drive.google.com/file/d/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz/edit'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = 'https://drive.google.com/file/d/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz/view?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/1Ab2Cd3EfGhIjKlMnOpQrStUvWxYz'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    module_0.download(var_0, filename=var_0, extract=var_0, progress=var_0)