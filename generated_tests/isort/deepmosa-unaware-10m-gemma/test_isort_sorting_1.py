# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.sorting as module_0
import enum as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'SX]_.!I@<}>l3b'
    module_0.module_key(var_0, var_0, section_name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'SX]_.!I@<}>l3b'
    module_0.module_key(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\x0c\xd32\xf9\xe1O'
    module_0.naturally(var_0)

def test_case_4():
    var_0 = "'\\7_f`("
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

def test_case_5():
    var_0 = "'\\_/fz("
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = "B.g6'0p4z\\qj|lVq"
    module_0.module_key(var_0, var_0, ignore_case=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_1.auto
    module_0.naturally(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = 'n3}B/'
    var_2 = [var_0, var_0, var_1]
    module_0.naturally(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '.,!x97W!}5msIM:"F'
    module_0.module_key(var_0, var_0)