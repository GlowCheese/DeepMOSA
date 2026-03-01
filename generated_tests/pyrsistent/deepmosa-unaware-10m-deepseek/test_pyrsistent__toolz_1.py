# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.get_in(var_0, var_0)

def test_case_1():
    var_0 = None
    var_1 = 3174
    var_2 = {var_1, var_0, var_1}
    var_3 = module_0.get_in(var_0, var_0, var_2)
    var_4 = None
    var_5 = module_0.get_in(var_4, var_4)
    var_6 = module_0.get_in(var_5, var_4, no_default=var_4)
    var_7 = module_0.get_in(var_4, var_4, no_default=var_5)
    var_8 = {var_5: var_6, var_6: var_6}
    with pytest.raises(TypeError):
        module_0.get_in(var_5, var_4, no_default=var_8)