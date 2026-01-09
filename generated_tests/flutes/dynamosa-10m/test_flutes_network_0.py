# Check out: https://github.com/GlowCheese/deepmosa
import flutes.network as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "|\r\\'a\n"
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.download(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "|\r\\'a\n"
    var_1 = None
    module_0.download(var_1, filename=var_0, extract=var_1, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'e8HjSQ'
    var_1 = None
    module_0.download(var_0, extract=var_1, progress=var_0)

def test_case_5():
    var_0 = ''
    var_1 = module_0.download(var_0, extract=var_0, progress=var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'lwGZsuC.'
    module_0.download(var_0, progress=var_0, bar_fn=var_0)