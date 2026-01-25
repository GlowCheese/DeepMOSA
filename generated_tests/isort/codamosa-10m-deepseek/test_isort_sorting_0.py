# Check out: https://github.com/GlowCheese/deepmosa
import isort.sorting as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '$\x0c3~I0_x-2@V]Z'
    module_0.module_key(var_0, var_0, ignore_case=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '~NB'
    module_0.module_key(var_0, var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.naturally(var_0)

def test_case_4():
    var_0 = '~NB'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

def test_case_5():
    var_0 = 'E}0\x0bT;~'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'TaI,=m_gh6.!'
    var_1 = None
    var_2 = True
    module_0.module_key(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'\x01\xce'
    module_0.naturally(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '.'
    var_1 = {var_0}
    var_2 = module_0.naturally(var_1)
    assert module_0.TYPE_CHECKING is False
    var_3 = None
    module_0.module_key(var_0, var_3, ignore_case=var_3, section_name=var_3)