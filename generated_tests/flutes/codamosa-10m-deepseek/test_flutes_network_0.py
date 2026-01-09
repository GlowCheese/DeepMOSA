# Check out: https://github.com/GlowCheese/deepmosa
import urllib.request as module_1

import flutes.network as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "|\r\\'a\n"
    var_1 = None
    module_0.download(var_0, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '3YG6'
    module_0.download(var_0, filename=var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = ''
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    module_0.download(var_0, var_1, progress=var_0, bar_fn=var_0, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1.localhost()
    assert var_0 == '127.0.0.1'
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.localhost()
    assert var_0 == '127.0.0.1'
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.download(var_0, progress=var_0, bar_fn=var_0)