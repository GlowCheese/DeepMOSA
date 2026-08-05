# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.exception as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.register_ipython_excepthook()
    var_1 = None
    var_2 = module_0.register_ipython_excepthook(var_1)
    var_3 = module_0.exception_wrapper(var_1)
    var_4 = module_0.register_ipython_excepthook(var_3)
    var_5 = module_0.exception_wrapper()
    var_3.require_parens(var_1, var_4)

def test_case_1():
    var_0 = module_0.register_ipython_excepthook()

def test_case_2():
    var_0 = None
    var_1 = module_0.log_exception(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ')&b42vx?L%\rd'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    module_0.log_exception(var_0, **var_1)

def test_case_4():
    var_0 = module_0.exception_wrapper()
    var_1 = module_0.log_exception(var_0, var_0)

def test_case_5():
    pass

def test_case_6():
    var_0 = module_0.exception_wrapper()