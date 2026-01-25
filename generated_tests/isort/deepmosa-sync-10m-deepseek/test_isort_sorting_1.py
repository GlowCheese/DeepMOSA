# Check out: https://github.com/GlowCheese/deepmosa
import isort.sorting as module_0
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'odul`e'
    module_0.module_key(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Modue'
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = []
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False
    var_2 = bool(var_1 == [])
    assert var_2 is True
    var_3 = 'cVE@]U3'
    var_4 = False
    module_0.module_key(var_3, var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = 'n3}B/'
    var_2 = [var_0, var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    var_5 = True
    var_6 = None
    var_7 = None
    module_0.module_key(var_6, var_6, var_7, var_6, var_5)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'x'
    var_1 = 'a10'
    var_2 = 'b2'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda x: x[var_4:]
    module_0.naturally(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = True
    var_1 = False
    var_2 = '.\x0bXjNb57@'
    var_3 = None
    module_0.module_key(var_2, var_3, ignore_case=var_0, straight_import=var_1)