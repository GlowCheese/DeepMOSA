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
    var_0 = '#^k\\hn1Dx6d'
    module_0.module_key(var_0, var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.naturally(var_0)

def test_case_3():
    var_0 = 'ghMQbCFFp(YVeroA'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

def test_case_4():
    var_0 = '+E6~GtN:'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '1xc+}f{\x0b\x0bz_83'
    module_0.module_key(var_0, var_0, ignore_case=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'AW>:Om\rkEg9#Qx'
    module_0.module_key(var_0, var_0, var_0, section_name=var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '(+[4mE2q'
    var_1 = {var_0, var_0}
    var_2 = module_0.naturally(var_1)
    assert module_0.TYPE_CHECKING is False
    var_3 = ()
    var_4 = module_0.naturally(var_3, var_3)
    var_5 = None
    module_0.module_key(var_0, var_5, section_name=var_4)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '.Kh*l'
    module_0.module_key(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '(+[4mEpq'
    var_1 = module_1.escape(var_0)
    assert var_1 == '\\(\\+\\[4mEpq'
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
    var_2 = False
    module_0.naturally(var_1, var_1, var_2)