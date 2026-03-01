# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = module_0.chunk(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 3
    var_6 = 4
    var_7 = [var_0, var_2, var_5, var_6]
    var_8 = module_0.chunk(var_7, var_0)
    var_9 = list(var_8)
    var_10 = [var_0, var_2, var_5, var_6]
    var_11 = module_0.chunk(var_10, var_2)
    var_12 = list(var_11)
    var_13 = 5
    var_14 = [var_0, var_2, var_5, var_6, var_13]
    var_15 = module_0.chunk(var_14, var_2)
    var_16 = list(var_15)
    var_17 = 6
    var_18 = 8
    var_19 = [var_0, var_2, var_5, var_6, var_13, var_17, var_9, var_18]
    var_20 = module_0.chunk(var_19, var_5)
    var_21 = list(var_20)