# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = []
    var_2 = 5
    var_3 = module_0.chunk(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = [var_0, var_5, var_6, var_7]
    var_9 = module_0.chunk(var_8, var_5)
    var_10 = list(var_9)
    var_11 = [var_0, var_5, var_6, var_7, var_2]
    var_12 = module_0.chunk(var_11, var_5)
    var_13 = list(var_12)
    var_14 = [var_0, var_5, var_6]
    var_15 = 10
    var_16 = module_0.chunk(var_14, var_15)
    var_17 = list(var_16)
    var_18 = module_0.identity(var_15)
    assert var_18 == 10
    var_19 = 'test'
    var_20 = module_0.identity(var_19)
    assert var_20 == 'test'