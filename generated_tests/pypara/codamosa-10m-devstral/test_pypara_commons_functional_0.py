# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = 3
    var_2 = 4
    var_3 = 5
    var_4 = 6
    var_5 = 7
    var_6 = 8
    var_7 = [var_0, var_5, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.chunk(var_7, var_1)
    var_9 = list(var_8)