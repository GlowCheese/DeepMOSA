# Check out: https://github.com/GlowCheese/deepmosa
import re as module_1

import isort.sorting as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '59+I%SBGp()'
    module_0.module_key(var_0, var_0, var_0, section_name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '/>%8YsO \x0b'
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.naturally(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = 'n3}B/'
    var_2 = [var_0, var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    var_5 = None
    var_6 = True
    module_0.module_key(var_1, var_5, var_6)

def test_case_5():
    var_0 = 'k\x0bG\x0c`'
    var_1 = [var_0, var_0, var_0]
    var_2 = False
    var_3 = module_0.naturally(var_1, reverse=var_2)
    assert module_0.TYPE_CHECKING is False
    var_4 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'HYuvfq\nRw\nH@B@cb'
    module_0.module_key(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.naturally(var_1, reverse=var_2)
    assert module_0.TYPE_CHECKING is False
    var_4 = None
    var_5 = '.7RhK^[`\x0b\\gz!|-.,7'
    module_0.module_key(var_5, var_2, ignore_case=var_4)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = '3 fWBBx$<D9=-n+v#'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    module_0.naturally(var_4, var_4)