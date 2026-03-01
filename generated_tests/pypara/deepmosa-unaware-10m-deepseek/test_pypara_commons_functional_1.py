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
    var_13 = 4
    var_14 = [var_1, var_8, var_9, var_13]
    var_15 = module_0.chunk(var_14, var_8)
    var_16 = list(var_15)
    var_17 = 6
    var_18 = [var_1, var_8, var_9, var_13, var_5, var_17]
    var_19 = module_0.chunk(var_18, var_9)
    var_20 = list(var_19)
    var_21 = list(var_12)
    var_22 = 7
    var_23 = [var_1, var_8, var_9, var_13, var_5, var_17, var_22]
    var_24 = module_0.chunk(var_23, var_9)
    var_25 = list(var_24)
    var_26 = [var_1, var_8, var_9]
    var_27 = module_0.chunk(var_26, var_5)
    var_28 = list(var_27)
    var_29 = [var_1, var_8, var_9]
    var_30 = module_0.chunk(var_29, var_9)
    var_31 = list(var_30)
    var_32 = module_0.chunk(var_4, var_8)
    var_33 = list(var_32)
    var_34 = (var_1, var_8)
    var_35 = (var_9, var_13)
    var_36 = (var_5, var_17)
    var_37 = [var_34, var_35, var_36]
    var_38 = module_0.chunk(var_37, var_8)
    var_39 = list(var_38)
    var_40 = [var_1, var_8, var_9, var_13]
    var_41 = module_0.chunk(var_40, var_8)
    var_42 = next(var_41)
    var_43 = next(var_41)