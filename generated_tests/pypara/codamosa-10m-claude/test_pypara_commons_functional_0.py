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
    var_10 = [var_1, var_5, var_8, var_9]
    var_11 = module_0.chunk(var_10, var_1)
    var_12 = list(var_11)
    var_13 = [var_1, var_5, var_8, var_9]
    var_14 = module_0.chunk(var_13, var_5)
    var_15 = list(var_14)
    var_16 = 5
    var_17 = [var_1, var_5, var_8, var_9, var_16]
    var_18 = module_0.chunk(var_17, var_5)
    var_19 = list(var_18)
    var_20 = [var_1, var_5, var_8]
    var_21 = module_0.chunk(var_20, var_16)
    var_22 = list(var_21)
    var_23 = [var_1, var_5, var_8]
    var_24 = module_0.chunk(var_23, var_8)
    var_25 = list(var_24)
    var_26 = [var_1]
    var_27 = module_0.chunk(var_26, var_1)
    var_28 = list(var_27)
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = 'd'
    var_33 = [var_29, var_30, var_31, var_32]
    var_34 = module_0.chunk(var_33, var_5)
    var_35 = list(var_34)
    var_36 = 7
    var_37 = 8
    var_38 = 9
    var_39 = [var_1, var_5, var_8, var_9, var_16, var_16, var_36, var_37, var_38]
    var_40 = module_0.chunk(var_39, var_8)
    var_41 = list(var_40)
    var_42 = 10
    var_43 = [var_1, var_5, var_8, var_9, var_16, var_35, var_36, var_37, var_38, var_42]
    var_44 = module_0.chunk(var_43, var_8)
    var_45 = list(var_44)