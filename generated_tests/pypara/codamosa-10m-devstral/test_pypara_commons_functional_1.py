# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = []
    var_1 = 1
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = 2
    var_6 = module_0.chunk(var_4, var_5)
    var_7 = list(var_6)
    var_8 = 3
    var_9 = 4
    var_10 = module_0.chunk(var_0, var_1)
    var_11 = list(var_10)
    var_12 = [var_1, var_5, var_8, var_9]
    var_13 = module_0.chunk(var_12, var_5)
    var_14 = list(var_13)
    var_15 = 5
    var_16 = [var_1, var_5, var_8, var_9, var_15]
    var_17 = module_0.chunk(var_16, var_5)
    var_18 = list(var_17)
    var_19 = 6
    var_20 = 7
    var_21 = module_0.chunk(var_14, var_8)
    var_22 = list(var_21)
    var_23 = 8
    var_24 = [var_1, var_5, var_8, var_9, var_15, var_19, var_20, var_23]
    var_25 = module_0.chunk(var_24, var_8)
    var_26 = list(var_25)