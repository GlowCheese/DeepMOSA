# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.exception as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    var_1 = module_0.register_ipython_excepthook()
    var_2 = module_0.register_ipython_excepthook(var_0)
    var_3 = module_0.log_exception(var_1, var_2)
    var_4 = module_0.exception_wrapper()
    var_4.__or__(var_3)

def test_case_1():
    var_0 = module_0.register_ipython_excepthook()

def test_case_2():
    var_0 = None
    var_1 = module_0.log_exception(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'another test error'
    var_1 = {var_0: var_0, var_0: var_0}
    module_0.log_exception(var_0, **var_1)

def test_case_4():
    pass

def test_case_5():
    var_0 = module_0.exception_wrapper()

def test_case_6():
    var_0 = 'PoolWorker'
    var_1 = module_0.log_exception(var_0, var_0)