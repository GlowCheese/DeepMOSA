# Check out: https://github.com/GlowCheese/deepmosa
import flutes.network as module_0
import pytest


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
    var_0 = None
    var_1 = ''
    var_2 = {}
    var_3 = module_0.download(var_1, bar_fn=var_0, **var_2)
    assert var_3 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'