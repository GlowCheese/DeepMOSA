# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.exception as module_0
import re as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    var_1 = module_0.register_ipython_excepthook(var_0)
    var_2 = None
    var_3 = 'y%V/L[h;\x0b'
    var_4 = {var_3: var_2, var_3: var_1}
    module_0.log_exception(var_2, **var_4)

def test_case_1():
    var_0 = module_0.register_ipython_excepthook()

def test_case_2():
    var_0 = None
    var_1 = module_0.log_exception(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = "ARmBK?Ub?@+79O8K'#"
    var_2 = {var_1: var_0, var_1: var_0}
    module_0.log_exception(var_0, **var_2)

def test_case_4():
    var_0 = module_1.RegexFlag.DEBUG
    var_1 = module_0.log_exception(var_0, var_0)
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

def test_case_5():
    var_0 = module_0.exception_wrapper()