# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'fU/:R24EZS]'
    module_0.download(var_0, filename=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ' 5r4n^]%L O9LR_&.1'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    module_0.download(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '5r4n]%L Oo9LR_&.1'
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_6():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/file/dxtest_file_id/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/tps:'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/file/d/test_file_id/view'
    module_0.download(var_0, filename=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/filed/test_file_id/view'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, var_1, var_1)
    assert var_2 == '/tmp/tps:'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.download(var_0, filename=var_0, progress=var_0)