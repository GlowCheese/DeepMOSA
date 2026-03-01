# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.chunk(var_4, var_0)
    var_6 = list(var_5)
    var_7 = [var_0, var_1, var_2, var_3]
    var_8 = module_0.chunk(var_7, var_1)
    var_9 = list(var_8)
    var_10 = 7
    var_11 = module_0.identity(var_6)
    with pytest.raises(TypeError):
        var_12 = list(var_10)