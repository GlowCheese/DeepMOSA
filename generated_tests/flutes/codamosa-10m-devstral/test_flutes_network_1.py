# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, extract=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '@Ww1l_{sh\tF\roQ\r7'
    module_0.download(var_0, extract=var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '@Ww1l_{sh\tF\roQ\r7'
    module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/123456789view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/123456789view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/file/23456789vie'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/tps:'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/123456789view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/123456789view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = module_1.exists(var_0)
    assert var_2 is False
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/file/d/13456789view'
    module_0.download(var_0, filename=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https://drive.google.com/file/d/13456789view'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, progress=var_0)
    assert var_2 == '/tmp/13456789view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_1.exists(var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'https://drive.googl4.com/file/d/123\\5689view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/123\\5689view'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_1.exists(var_2)