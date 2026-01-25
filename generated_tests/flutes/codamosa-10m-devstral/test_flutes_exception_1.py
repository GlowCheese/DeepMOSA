# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.exception as module_0
import ast as module_1

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
    var_0 = None
    var_1 = "5\t#6~gcEe;;rF'oro"
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    module_0.log_exception(var_0, **var_2)

def test_case_4():
    var_0 = module_0.exception_wrapper()

def test_case_5():
    var_0 = module_1._Precedence.ATOM
    var_1 = module_0.log_exception(var_0, var_0)
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096