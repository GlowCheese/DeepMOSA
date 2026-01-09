# Check out: https://github.com/GlowCheese/deepmosa
import flutes.exception as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = "8nEi)9/)#?'?Z5i"
    var_1 = {var_0: var_0}
    var_2 = module_0.register_ipython_excepthook(var_0)
    module_0.log_exception(var_0, var_0, **var_1)

def test_case_1():
    var_0 = module_0.register_ipython_excepthook()

def test_case_2():
    var_0 = None
    var_1 = module_0.log_exception(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '6V1UrYuc]`qT_@w('
    var_1 = {var_0: var_0}
    module_0.log_exception(var_1, **var_1)

def test_case_4():
    var_0 = 'EfXuD'
    var_1 = module_0.log_exception(var_0, var_0)

def test_case_5():
    var_0 = module_0.exception_wrapper()