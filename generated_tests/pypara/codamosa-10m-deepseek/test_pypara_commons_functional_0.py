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
    var_7 = module_0.chunk(var_4, var_0)
    var_8 = list(var_7)
    var_9 = [var_0, var_2, var_5, var_6]
    var_10 = module_0.chunk(var_9, var_2)
    var_11 = list(var_10)
    var_12 = 5
    var_13 = [var_0, var_2, var_5, var_6, var_12]
    var_14 = module_0.chunk(var_13, var_2)
    var_15 = list(var_14)
    var_16 = 6
    var_17 = [var_0, var_2, var_5, var_6, var_12, var_16]
    var_18 = module_0.chunk(var_17, var_5)
    var_19 = list(var_18)
    var_20 = [var_0]
    var_21 = module_0.chunk(var_20, var_5)
    var_22 = list(var_21)
    var_23 = 7
    var_24 = [var_0, var_2, var_5, var_6, var_12, var_16, var_23]
    var_25 = module_0.chunk(var_24, var_6)
    var_26 = list(var_25)