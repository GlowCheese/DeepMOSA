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
    var_10 = 4
    var_11 = [var_1, var_8, var_9, var_10]
    var_12 = module_0.chunk(var_11, var_8)
    var_13 = list(var_12)
    var_14 = 6
    var_15 = [var_1, var_8, var_9, var_10, var_5, var_14]
    var_16 = module_0.chunk(var_15, var_9)
    var_17 = list(var_16)
    var_18 = [var_1, var_8, var_9, var_10, var_5]
    var_19 = module_0.chunk(var_18, var_8)
    var_20 = list(var_19)
    var_21 = 7
    var_22 = [var_1, var_8, var_9, var_10, var_5, var_14, var_21]
    var_23 = module_0.chunk(var_22, var_9)
    var_24 = list(var_23)
    var_25 = [var_1, var_8, var_9]
    var_26 = module_0.chunk(var_25, var_5)
    var_27 = list(var_26)
    var_28 = [var_1]
    var_29 = 10
    var_30 = module_0.chunk(var_28, var_29)
    var_31 = list(var_30)
    var_32 = [var_1, var_8, var_9]
    var_33 = module_0.chunk(var_32, var_1)
    var_34 = list(var_33)
    var_35 = 'a'
    var_36 = 'b'
    var_37 = 'c'
    var_38 = [var_35, var_36, var_37, var_31]
    var_39 = module_0.chunk(var_38, var_8)
    var_40 = list(var_39)
    var_41 = 1.1
    var_42 = 2.2
    var_43 = 3.3
    var_44 = 4.4
    var_45 = [var_41, var_42, var_43, var_44]
    var_46 = module_0.chunk(var_45, var_8)
    var_47 = list(var_46)
    var_48 = [var_1, var_8, var_9, var_10]
    var_49 = module_0.chunk(var_48, var_8)
    var_50 = list(var_49)