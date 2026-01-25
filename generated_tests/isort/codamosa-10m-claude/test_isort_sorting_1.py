# Check out: https://github.com/GlowCheese/deepmosa
import isort.sorting as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'X)$m15K\r'
    module_0.module_key(var_0, var_0, ignore_case=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'ka|I_vlQe604f{\x0bk'
    module_0.module_key(var_0, var_0, var_0, section_name=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.naturally(var_0)

def test_case_5():
    var_0 = 'zl[4tY;U.o'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'\xae\x87:K\x7f\x9e_\xd7\xdf40g\xabp\x99\xff4\x13t'
    module_0.naturally(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '.'
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = '3 fWBBx$<D9=-n+v#'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    module_0.naturally(var_4, var_4)