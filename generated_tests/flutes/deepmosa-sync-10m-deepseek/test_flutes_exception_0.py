# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.exception as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.register_ipython_excepthook(var_0)
    var_2 = 'ls'
    var_3 = module_0.log_exception(var_1, var_2)

def test_case_1():
    var_0 = module_0.register_ipython_excepthook()

def test_case_2():
    var_0 = None
    var_1 = module_0.log_exception(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '|Ghu1":av}-8JVY\'0g#'
    var_1 = {var_0: var_0, var_0: var_0}
    module_0.log_exception(var_1, **var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'Y[g'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    module_0.log_exception(var_0, var_0, **var_1)

def test_case_5():
    var_0 = module_0.exception_wrapper()