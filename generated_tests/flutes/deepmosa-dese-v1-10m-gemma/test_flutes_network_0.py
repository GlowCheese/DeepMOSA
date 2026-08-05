# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import email._encoded_words as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '\ned;;d*5#dl=fq'
    module_0.download(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'GhfpUR\x0b\x0b0vY'
    module_0.download(var_0, var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'GhpUR\x0b\x0b0vY'
    module_0.download(var_0, extract=var_0, progress=var_0)

def test_case_4():
    var_0 = 'fil.d_x'
    var_1 = module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)
    assert var_1 == '/tmp/fil.d_x'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_5():
    var_0 = 'http://examble.com'
    var_1 = 'fil.d_x'
    var_2 = module_0.download(var_0, filename=var_1, extract=var_1, bar_fn=var_1)
    assert var_2 == '/tmp/fil.d_x'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'https://drive.google.com/d/xuz-789/edit#gid=0'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/xuz-789'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_1.len_q(var_1)

def test_case_7():
    var_0 = 'https://drive.google.com/d/1abc123/view'
    var_1 = None
    var_2 = module_0.download(var_0, progress=var_0, bar_fn=var_1)
    assert var_2 == '/tmp/1abc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0._extract_google_drive_file_id(var_0)
    assert var_3 == '1abc123'
    var_4 = 'https://drive.google.com/d/xyz-789/edit#gid=0'
    var_5 = module_0._extract_google_drive_file_id(var_4)
    assert var_5 == 'xyz-789'