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
    var_5 = 5
    var_6 = module_0.chunk(var_4, var_5)
    var_7 = list(var_6)
    var_8 = 2
    var_9 = 3
    var_10 = [var_1, var_8, var_9]
    var_11 = module_0.chunk(var_10, var_1)
    var_12 = list(var_11)
    var_13 = module_0.chunk(var_12, var_8)
    var_14 = list(var_13)
    var_15 = list(var_10)
    var_16 = [var_1, var_8]
    var_17 = 10
    var_18 = module_0.chunk(var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'a'
    var_21 = 'c'
    var_22 = [var_20, var_12, var_21]
    var_23 = module_0.chunk(var_22, var_8)
    var_24 = list(var_23)