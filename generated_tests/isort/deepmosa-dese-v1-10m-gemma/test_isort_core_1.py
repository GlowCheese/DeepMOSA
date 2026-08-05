# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import urllib.request as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.localhost()
    assert var_0 == '127.0.0.1'
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 13
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.process(var_0, var_0, raise_on_skip=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\r'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 13
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    var_1 = module_1.localhost()
    assert var_1 == '127.0.0.1'
    var_2 = var_1.__repr__()
    assert var_2 == "'127.0.0.1'"
    module_0.process(var_2, var_1, var_2, var_0)