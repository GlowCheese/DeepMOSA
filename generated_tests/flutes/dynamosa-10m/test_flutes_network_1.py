# Check out: https://github.com/GlowCheese/deepmosa
import urllib.request as module_1

import flutes.network as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '":z}F Pc!K5)H'
    var_1 = None
    var_2 = {}
    module_0.download(var_0, var_1, progress=var_0, bar_fn=var_1, **var_2)

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
    var_0 = 'mT0f!.e'
    module_0.download(var_0, var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = {}
    module_0.download(var_1, filename=var_1, extract=var_1, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.thishost()
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, filename=var_0, extract=var_0, progress=var_0, bar_fn=var_0)

def test_case_6():
    var_0 = ''
    var_1 = None
    var_2 = module_0.download(var_1, filename=var_0, bar_fn=var_1)
    assert var_2 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'