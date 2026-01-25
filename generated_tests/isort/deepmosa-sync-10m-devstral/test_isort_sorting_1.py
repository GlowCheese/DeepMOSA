# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.sorting as module_0
import builtins as module_1

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'Module'
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "'SMmpzkHAOZ}"
    module_0.module_key(var_0, var_0, var_0, section_name=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\x0bnx\x0c)\rO]'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False
    module_0.module_key(var_0, var_0)

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
    var_0 = 'oule'
    module_0.module_key(var_0, var_0, ignore_case=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'import Os'
    var_1 = module_1.BaseException
    module_0.naturally(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '.cj1Sz@gN6'
    var_1 = None
    var_2 = True
    module_0.module_key(var_0, var_1, var_1, var_2)