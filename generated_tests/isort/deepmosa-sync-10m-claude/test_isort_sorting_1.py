# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.sorting as module_0

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ''
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ':St\\05S34&lXY2'
    module_0.module_key(var_0, var_0, var_0)

def test_case_3():
    var_0 = []
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

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
    var_0 = '9'
    module_0.module_key(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = True
    module_0.naturally(var_1, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '...module'
    var_1 = None
    module_0.module_key(var_0, var_1, straight_import=var_1)