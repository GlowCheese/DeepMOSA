# Check out: https://github.com/GlowCheese/deepmosa
import isort.sorting as module_0
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ''
    module_0.module_key(var_0, var_0, section_name=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '7Xyl'
    module_0.module_key(var_0, var_0, var_0, section_name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.naturally(var_0, var_0)

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
    var_0 = '*zmlh|dx=\n\rb{R$'
    module_0.module_key(var_0, var_0, ignore_case=var_0, section_name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = 'n3}B/'
    var_2 = [var_0, var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    module_0.naturally(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '.\nIV,6,X+;l#`gEA'
    var_1 = None
    module_0.module_key(var_0, var_1, straight_import=var_1)