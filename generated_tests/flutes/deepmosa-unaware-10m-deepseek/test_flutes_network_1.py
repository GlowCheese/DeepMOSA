# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'g_t9/}~A33=/,*wBNs4'
    module_0.download(var_0, progress=var_0)

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
    var_0 = 'V:z}k Pc!O)H'
    module_0.download(var_0, var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'Ko},lm/'
    module_0.download(var_0, filename=var_0, progress=var_0, bar_fn=var_0)

def test_case_6():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'