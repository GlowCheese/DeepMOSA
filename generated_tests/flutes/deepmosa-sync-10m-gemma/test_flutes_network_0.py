# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/filed/est_id_789/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/tps:'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'https://drive.google.com/file/d/test_id_123/view'
    module_0.download(var_0, filename=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = '_$:\\'
    module_0.download(var_1, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = "*`{!\rO'a@@9'3"
    var_2 = {var_1: var_0, var_1: var_0}
    module_0.download(var_0, filename=var_1, progress=var_2, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = 'G0hfpUR\x0b\x0b0vY'
    module_0.download(var_1, var_0, progress=var_1, bar_fn=var_0)

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/my_special_id/view'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, progress=var_1)
    assert var_2 == '/tmp/my_special_id'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/d/1abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '1abc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'