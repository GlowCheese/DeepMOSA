####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_36, var_37]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 6
    var_51 = [var_37, var_15]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 7
    var_53 = [var_15, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_15, var_38]
    var_56 = 'Not Found'
    var_57 = module_0.get_in(var_55, var_48, var_56)
    assert var_57 == 'Not Found'
    var_58 = 'a'
    var_59 = 'd'
    var_60 = 'b'
    var_61 = {var_60: var_36}
    var_62 = 'c'
    var_63 = {var_62: var_37}
    var_64 = [var_61, var_63]
    var_65 = 'e'
    var_66 = [var_38, var_40, var_41]
    var_67 = {var_65: var_66}
    var_68 = {var_58: var_64, var_59: var_67}
    var_69 = [var_58, var_15, var_60]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 == 1
    var_71 = [var_59, var_65, var_36]
    var_72 = module_0.get_in(var_71, var_68)
    assert var_72 == 4
    var_73 = [var_58, var_36, var_59]
    var_74 = module_0.get_in(var_73, var_68)
    assert var_74 is None
    var_75 = [var_59, var_65, var_41]
    var_76 = 'Out of range'
    var_77 = module_0.get_in(var_75, var_68, var_76)
    assert var_77 == 'Out of range'
    var_78 = []
    var_79 = {var_58: var_36}
    var_80 = module_0.get_in(var_78, var_79)
    var_81 = 'x'
    var_82 = [var_81]
    var_83 = {}
    var_84 = 'Default'
    var_85 = module_0.get_in(var_82, var_83, var_84)
    assert var_85 == 'Default'
    var_86 = 'All tests passed!'
    var_87 = print(var_86)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_38, var_21, var_9]
    var_54 = module_0.get_in(var_53, var_46, var_13)
    assert var_54 == 0
    var_55 = 'All tests passed!'
    var_56 = print(var_55)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = 2
    var_12 = module_0.get_in(var_10, var_6, var_11)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'd'
    var_16 = [var_13, var_14, var_15]
    var_17 = True
    var_18 = module_0.get_in(var_16, var_6, no_default=var_17)
    var_19 = 3
    var_20 = [var_16, var_11, var_19]
    var_21 = 4
    var_22 = 5
    var_23 = 6
    var_24 = [var_21, var_22, var_23]
    var_25 = [var_20, var_24]
    var_26 = 0
    var_27 = [var_26, var_16]
    var_28 = module_0.get_in(var_27, var_25)
    assert var_28 == 2
    var_29 = [var_16, var_11]
    var_30 = module_0.get_in(var_29, var_25)
    assert var_30 == 6
    var_31 = [var_11, var_26]
    var_32 = None
    var_33 = module_0.get_in(var_31, var_25, var_32)
    assert var_33 is None
    var_34 = {var_14: var_16}
    var_35 = {var_15: var_11}
    var_36 = [var_34, var_35]
    var_37 = {var_13: var_36}
    var_38 = [var_13, var_26, var_14]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 1
    var_40 = [var_13, var_16, var_15]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 2
    var_42 = 'x'
    var_43 = 10
    var_44 = {var_42: var_43}
    var_45 = 'y'
    var_46 = [var_45]
    var_47 = 20
    var_48 = module_0.get_in(var_46, var_44, var_47)
    assert var_48 == 20
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_36, var_37]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 6
    var_51 = [var_37, var_15]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 7
    var_53 = [var_15, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_15, var_38]
    var_56 = -1
    var_57 = module_0.get_in(var_55, var_48, var_56)
    assert var_57 == -1
    var_58 = 'a'
    var_59 = 'd'
    var_60 = 'b'
    var_61 = {var_60: var_36}
    var_62 = 'c'
    var_63 = {var_62: var_37}
    var_64 = [var_61, var_63]
    var_65 = 'e'
    var_66 = [var_38, var_40, var_41]
    var_67 = {var_65: var_66}
    var_68 = {var_58: var_64, var_59: var_67}
    var_69 = [var_58, var_15, var_60]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 == 1
    var_71 = [var_59, var_65, var_37]
    var_72 = module_0.get_in(var_71, var_68)
    assert var_72 == 5
    var_73 = [var_58, var_36, var_59]
    var_74 = module_0.get_in(var_73, var_68)
    assert var_74 is None
    var_75 = [var_58, var_36, var_59]
    var_76 = module_0.get_in(var_75, var_68, var_15)
    assert var_76 == 0
    var_77 = 'All tests passed!'
    var_78 = print(var_77)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = 2
    var_12 = module_0.get_in(var_10, var_6, var_11)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'd'
    var_16 = [var_13, var_14, var_15]
    var_17 = True
    var_18 = module_0.get_in(var_16, var_6, no_default=var_17)
    var_19 = 3
    var_20 = [var_16, var_11, var_19]
    var_21 = 4
    var_22 = 5
    var_23 = 6
    var_24 = [var_21, var_22, var_23]
    var_25 = [var_20, var_24]
    var_26 = 0
    var_27 = [var_26, var_16]
    var_28 = module_0.get_in(var_27, var_25)
    assert var_28 == 2
    var_29 = [var_16, var_11]
    var_30 = module_0.get_in(var_29, var_25)
    assert var_30 == 6
    var_31 = [var_11, var_26]
    var_32 = None
    var_33 = module_0.get_in(var_31, var_25, var_32)
    assert var_33 is None
    var_34 = {var_14: var_16}
    var_35 = {var_15: var_11}
    var_36 = [var_34, var_35]
    var_37 = {var_13: var_36}
    var_38 = [var_13, var_26, var_14]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 1
    var_40 = [var_13, var_16, var_15]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 2
    var_42 = 'All tests passed!'
    var_43 = print(var_42)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = 10
    var_34 = [var_26, var_29, var_33]
    var_35 = module_0.get_in(var_34, var_14)
    assert var_35 is None
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = 4
    var_20 = 5
    var_21 = 6
    var_22 = [var_19, var_20, var_21]
    var_23 = [var_18, var_22]
    var_24 = [var_13, var_15]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 2
    var_26 = [var_15, var_16]
    var_27 = module_0.get_in(var_26, var_23)
    assert var_27 == 6
    var_28 = [var_16, var_13]
    var_29 = module_0.get_in(var_28, var_23)
    assert var_29 is None
    var_30 = {var_1: var_15}
    var_31 = {var_2: var_16}
    var_32 = [var_30, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_13, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 1
    var_36 = [var_0, var_15, var_2]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 == 2
    var_38 = {var_1: var_15}
    var_39 = {var_0: var_38}
    var_40 = 'a'
    var_41 = 'c'
    var_42 = [var_40, var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_39, no_default=var_43)
    var_45 = 'All tests passed!'
    var_46 = print(var_45)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_38, var_21, var_9]
    var_54 = module_0.get_in(var_53, var_46, var_13)
    assert var_54 == 0
    var_55 = 'All tests passed!'
    var_56 = print(var_55)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_21]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 is None
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_26, var_13)
    assert var_32 == 0
    var_33 = 1
    var_34 = 2
    var_35 = [var_33, var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_26, no_default=var_36)
    var_38 = {var_34: var_36}
    var_39 = {var_35: var_21}
    var_40 = [var_38, var_39]
    var_41 = {var_33: var_40}
    var_42 = [var_33, var_13, var_34]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 1
    var_44 = [var_33, var_36, var_35]
    var_45 = module_0.get_in(var_44, var_41)
    assert var_45 == 2
    var_46 = [var_33, var_21, var_9]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 is None
    var_48 = [var_33, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_41, var_13)
    assert var_49 == 0
    var_50 = 'All tests passed!'
    var_51 = print(var_50)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    assert var_9 == 42
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    assert var_12 is None
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    assert var_21 == 0
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_38, var_21, var_9]
    var_54 = module_0.get_in(var_53, var_46, var_13)
    assert var_54 == 0
    var_55 = 'a'
    var_56 = 2
    var_57 = 'd'
    var_58 = [var_55, var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_46, no_default=var_59)
    var_61 = []
    var_62 = {var_55: var_58}
    var_63 = module_0.get_in(var_61, var_62)
    var_64 = []
    var_65 = [var_58, var_21, var_22]
    var_66 = module_0.get_in(var_64, var_65)
    var_67 = 'x'
    var_68 = 'y'
    var_69 = 'z'
    var_70 = 42
    var_71 = {var_69: var_70}
    var_72 = {var_68: var_71}
    var_73 = {var_67: var_72}
    var_74 = [var_67, var_68, var_69]
    var_75 = 'w'
    var_76 = [var_67, var_68, var_75]
    var_77 = [var_67, var_68, var_75]
    var_78 = 0
    var_79 = 'x'
    var_80 = 'y'
    var_81 = 'w'
    var_82 = [var_79, var_80, var_81]
    var_83 = True
    var_84 = 'All tests passed!'
    var_85 = print(var_84)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_36, var_37]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 6
    var_51 = [var_37, var_15]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 7
    var_53 = [var_15, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_15, var_38]
    var_56 = 'Not Found'
    var_57 = module_0.get_in(var_55, var_48, var_56)
    assert var_57 == 'Not Found'
    var_58 = 'a'
    var_59 = 'd'
    var_60 = 'b'
    var_61 = {var_60: var_36}
    var_62 = 'c'
    var_63 = {var_62: var_37}
    var_64 = [var_61, var_63]
    var_65 = 'e'
    var_66 = [var_38, var_40, var_41]
    var_67 = {var_65: var_66}
    var_68 = {var_58: var_64, var_59: var_67}
    var_69 = [var_58, var_15, var_60]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 == 1
    var_71 = [var_59, var_65, var_36]
    var_72 = module_0.get_in(var_71, var_68)
    assert var_72 == 4
    var_73 = [var_58, var_36, var_59]
    var_74 = module_0.get_in(var_73, var_68)
    assert var_74 is None
    var_75 = [var_59, var_65, var_38]
    var_76 = 'Out of bounds'
    var_77 = module_0.get_in(var_75, var_68, var_76)
    assert var_77 == 'Out of bounds'
    var_78 = 'x'
    var_79 = [var_78]
    var_80 = {}
    var_81 = True
    var_82 = module_0.get_in(var_79, var_80, no_default=var_81)
    var_83 = 0
    var_84 = [var_83]
    var_85 = []
    var_86 = True
    var_87 = module_0.get_in(var_84, var_85, no_default=var_86)
    var_88 = 'All tests passed!'
    var_89 = print(var_88)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 2
    var_16 = 3
    var_17 = [var_3, var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = [var_18, var_19, var_20]
    var_22 = [var_17, var_21]
    var_23 = [var_13, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3, var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 6
    var_27 = [var_15, var_13]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = {var_1: var_3}
    var_30 = {var_2: var_15}
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_13, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 1
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 2
    var_37 = [var_0, var_15, var_9]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = {var_1: var_3}
    var_40 = {var_0: var_39}
    var_41 = 'a'
    var_42 = 'c'
    var_43 = [var_41, var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_40, no_default=var_44)
    var_46 = {var_42: var_44}
    var_47 = {var_41: var_46}
    var_48 = [var_41, var_43]
    var_49 = module_0.get_in(var_48, var_47, var_13)
    assert var_49 == 0
    var_50 = {var_41: var_44}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = []
    var_54 = {var_38: var_41}
    var_55 = module_0.get_in(var_53, var_54)
    var_56 = []
    var_57 = [var_41, var_21, var_22]
    var_58 = module_0.get_in(var_56, var_57)
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'd'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = []
    var_54 = {var_38: var_41}
    var_55 = module_0.get_in(var_53, var_54)
    var_56 = []
    var_57 = [var_41, var_21, var_22]
    var_58 = module_0.get_in(var_56, var_57)
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_15, var_36]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 2
    var_51 = [var_37, var_37]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 9
    var_53 = [var_36, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_38, var_15]
    var_56 = module_0.get_in(var_55, var_48)
    assert var_56 is None
    var_57 = [var_36, var_38]
    var_58 = 'default'
    var_59 = module_0.get_in(var_57, var_48, var_58)
    assert var_59 == 'default'
    var_60 = 'a'
    var_61 = 'd'
    var_62 = 'b'
    var_63 = {var_62: var_36}
    var_64 = 'c'
    var_65 = {var_64: var_37}
    var_66 = [var_63, var_65]
    var_67 = 'e'
    var_68 = [var_38, var_40, var_41]
    var_69 = {var_67: var_68}
    var_70 = {var_60: var_66, var_61: var_69}
    var_71 = [var_60, var_15, var_62]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 1
    var_73 = [var_61, var_67, var_36]
    var_74 = module_0.get_in(var_73, var_70)
    assert var_74 == 4
    var_75 = [var_60, var_37, var_62]
    var_76 = module_0.get_in(var_75, var_70)
    assert var_76 is None
    var_77 = 'f'
    var_78 = [var_61, var_77]
    var_79 = module_0.get_in(var_78, var_70)
    assert var_79 is None
    var_80 = 'All tests passed!'
    var_81 = print(var_80)



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = []
    var_54 = {var_38: var_41}
    var_55 = module_0.get_in(var_53, var_54)
    var_56 = []
    var_57 = [var_41, var_21, var_22]
    var_58 = module_0.get_in(var_56, var_57)
    var_59 = 'x'
    var_60 = 'y'
    var_61 = [var_59, var_60]
    var_62 = {}
    var_63 = 'default'
    var_64 = module_0.get_in(var_61, var_62, var_63)
    assert var_64 == 'default'
    var_65 = [var_13, var_41]
    var_66 = []
    var_67 = module_0.get_in(var_65, var_66, var_63)
    assert var_67 == 'default'
    var_68 = 'All tests passed!'
    var_69 = print(var_68)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_38, var_21, var_9]
    var_54 = module_0.get_in(var_53, var_46, var_13)
    assert var_54 == 0
    var_55 = 'a'
    var_56 = 2
    var_57 = 'd'
    var_58 = [var_55, var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_46, no_default=var_59)
    var_61 = 'All tests passed!'
    var_62 = print(var_61)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = [var_31, var_32, var_33]
    var_35 = 4
    var_36 = 5
    var_37 = 6
    var_38 = [var_35, var_36, var_37]
    var_39 = 7
    var_40 = 8
    var_41 = 9
    var_42 = [var_39, var_40, var_41]
    var_43 = [var_34, var_38, var_42]
    var_44 = [var_31, var_32]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 6
    var_46 = [var_32, var_15]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 7
    var_48 = [var_15, var_33]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = (var_31, var_32, var_33)
    var_51 = (var_35, var_36, var_37)
    var_52 = (var_39, var_40, var_41)
    var_53 = (var_50, var_51, var_52)
    var_54 = [var_31, var_32]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 == 6
    var_56 = [var_32, var_15]
    var_57 = module_0.get_in(var_56, var_53)
    assert var_57 == 7
    var_58 = [var_15, var_33]
    var_59 = module_0.get_in(var_58, var_53)
    assert var_59 is None
    var_60 = 'a'
    var_61 = 'c'
    var_62 = 'b'
    var_63 = {var_62: var_33}
    var_64 = [var_31, var_32, var_63]
    var_65 = [var_37, var_39]
    var_66 = (var_35, var_36, var_65)
    var_67 = {var_60: var_64, var_61: var_66}
    var_68 = [var_60, var_32, var_62]
    var_69 = module_0.get_in(var_68, var_67)
    assert var_69 == 3
    var_70 = [var_61, var_32, var_31]
    var_71 = module_0.get_in(var_70, var_67)
    assert var_71 == 7
    var_72 = [var_60, var_33]
    var_73 = module_0.get_in(var_72, var_67)
    assert var_73 is None
    var_74 = 'x'
    var_75 = 'y'
    var_76 = [var_74, var_75]
    var_77 = {}
    var_78 = 'default'
    var_79 = module_0.get_in(var_76, var_77, var_78)
    assert var_79 == 'default'
    var_80 = [var_74, var_75]
    var_81 = {}
    var_82 = None
    var_83 = module_0.get_in(var_80, var_81, var_82)
    assert var_83 is None
    var_84 = 'x'
    var_85 = 'y'
    var_86 = [var_84, var_85]
    var_87 = {}
    var_88 = True
    var_89 = module_0.get_in(var_86, var_87, no_default=var_88)
    var_90 = 'All tests passed!'
    var_91 = print(var_90)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_18, var_21, var_22]
    var_24 = 4
    var_25 = 5
    var_26 = 6
    var_27 = [var_24, var_25, var_26]
    var_28 = [var_23, var_27]
    var_29 = [var_13, var_18]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 6
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 is None
    var_35 = [var_21, var_13]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_28, var_36)
    assert var_37 == -1
    var_38 = 2
    var_39 = 0
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_28, no_default=var_41)
    var_43 = {var_39: var_41}
    var_44 = {var_40: var_21}
    var_45 = [var_43, var_44]
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_13, var_39]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 1
    var_49 = [var_38, var_41, var_40]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_38, var_21, var_9]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_38, var_21, var_9]
    var_54 = module_0.get_in(var_53, var_46, var_13)
    assert var_54 == 0
    var_55 = 'a'
    var_56 = 2
    var_57 = 'd'
    var_58 = [var_55, var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_46, no_default=var_59)
    var_61 = []
    var_62 = {var_55: var_58}
    var_63 = module_0.get_in(var_61, var_62)
    var_64 = []
    var_65 = [var_58, var_21, var_22]
    var_66 = module_0.get_in(var_64, var_65)
    var_67 = [var_58, var_21, var_22]
    var_68 = [var_58]
    var_69 = {var_56: var_58}
    var_70 = {var_55: var_69}
    var_71 = [var_55, var_56]
    var_72 = 'All tests passed!'
    var_73 = print(var_72)



# Parsed testcases at query #25
#--------------------------




####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_21]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 is None
    var_31 = [var_18, var_21]
    var_32 = module_0.get_in(var_31, var_26, var_13)
    assert var_32 == 0
    var_33 = 1
    var_34 = 2
    var_35 = [var_33, var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_26, no_default=var_36)
    var_38 = {var_34: var_36}
    var_39 = {var_35: var_21}
    var_40 = [var_38, var_39]
    var_41 = {var_33: var_40}
    var_42 = [var_33, var_13, var_34]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 1
    var_44 = [var_33, var_36, var_35]
    var_45 = module_0.get_in(var_44, var_41)
    assert var_45 == 2
    var_46 = [var_33, var_21, var_9]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 is None
    var_48 = [var_33, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_41, var_13)
    assert var_49 == 0
    var_50 = 'a'
    var_51 = 2
    var_52 = 'd'
    var_53 = [var_50, var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_41, no_default=var_54)
    var_56 = []
    var_57 = module_0.get_in(var_56, var_6)
    var_58 = []
    var_59 = module_0.get_in(var_58, var_26)
    var_60 = []
    var_61 = module_0.get_in(var_60, var_41)
    var_62 = 'x'
    var_63 = 'y'
    var_64 = 'z'
    var_65 = [var_62, var_63, var_64]
    var_66 = {}
    var_67 = 'not found'
    var_68 = module_0.get_in(var_65, var_66, var_67)
    assert var_68 == 'not found'
    var_69 = [var_13, var_53, var_21]
    var_70 = []
    var_71 = module_0.get_in(var_69, var_70, var_67)
    assert var_71 == 'not found'
    var_72 = 'All tests passed!'
    var_73 = print(var_72)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = 2
    var_12 = module_0.get_in(var_10, var_6, var_11)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'd'
    var_16 = [var_13, var_14, var_15]
    var_17 = True
    var_18 = module_0.get_in(var_16, var_6, no_default=var_17)
    var_19 = 3
    var_20 = [var_16, var_11, var_19]
    var_21 = 4
    var_22 = 5
    var_23 = 6
    var_24 = [var_21, var_22, var_23]
    var_25 = [var_20, var_24]
    var_26 = 0
    var_27 = [var_26, var_16]
    var_28 = module_0.get_in(var_27, var_25)
    assert var_28 == 2
    var_29 = [var_16, var_11]
    var_30 = module_0.get_in(var_29, var_25)
    assert var_30 == 6
    var_31 = [var_11, var_26]
    var_32 = None
    var_33 = module_0.get_in(var_31, var_25, var_32)
    assert var_33 is None
    var_34 = {var_14: var_16}
    var_35 = {var_15: var_11}
    var_36 = [var_34, var_35]
    var_37 = {var_13: var_36}
    var_38 = [var_13, var_26, var_14]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 1
    var_40 = [var_13, var_16, var_15]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 2
    var_42 = 'x'
    var_43 = 10
    var_44 = {var_42: var_43}
    var_45 = 'y'
    var_46 = [var_45]
    var_47 = 20
    var_48 = module_0.get_in(var_46, var_44, var_47)
    assert var_48 == 20
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = 2
    var_12 = module_0.get_in(var_10, var_6, var_11)
    assert var_12 == 2
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'd'
    var_16 = [var_13, var_14, var_15]
    var_17 = True
    var_18 = module_0.get_in(var_16, var_6, no_default=var_17)
    var_19 = [var_16, var_11]
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = 0
    var_25 = [var_24, var_16]
    var_26 = module_0.get_in(var_25, var_23)
    assert var_26 == 2
    var_27 = {var_14: var_16}
    var_28 = {var_15: var_11}
    var_29 = [var_27, var_28]
    var_30 = {var_13: var_29}
    var_31 = [var_13, var_24, var_14]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = 'All tests passed'
    var_34 = print(var_33)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = []
    var_34 = module_0.get_in(var_33, var_14)
    var_35 = 'apple'
    var_36 = [var_26, var_29, var_35]
    var_37 = module_0.get_in(var_36, var_14)
    assert var_37 is None
    var_38 = 10
    var_39 = [var_26, var_29, var_38]
    var_40 = module_0.get_in(var_39, var_14)
    assert var_40 is None
    var_41 = 'All tests passed!'
    var_42 = print(var_41)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 2
    var_16 = 3
    var_17 = [var_3, var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = [var_18, var_19, var_20]
    var_22 = [var_17, var_21]
    var_23 = [var_13, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3, var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 6
    var_27 = [var_15, var_13]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = {var_1: var_3}
    var_30 = {var_2: var_15}
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_13, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 1
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 2
    var_37 = [var_0, var_15, var_9]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = {var_1: var_3}
    var_40 = {var_0: var_39}
    var_41 = 'a'
    var_42 = 'c'
    var_43 = [var_41, var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_40, no_default=var_44)
    var_46 = {var_41: var_44}
    var_47 = []
    var_48 = module_0.get_in(var_47, var_46)
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 2
    var_16 = 3
    var_17 = [var_3, var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = [var_18, var_19, var_20]
    var_22 = [var_17, var_21]
    var_23 = [var_13, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3, var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 6
    var_27 = [var_15, var_13]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = {var_1: var_3}
    var_30 = {var_2: var_15}
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_13, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 1
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 2
    var_37 = {var_1: var_3}
    var_38 = {var_0: var_37}
    var_39 = 'a'
    var_40 = 'c'
    var_41 = [var_39, var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_38, no_default=var_42)
    var_44 = {var_40: var_42}
    var_45 = {var_39: var_44}
    var_46 = [var_39, var_41]
    var_47 = module_0.get_in(var_46, var_45, var_13)
    assert var_47 == 0
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = -1
    var_35 = module_0.get_in(var_33, var_26, var_34)
    assert var_35 == -1
    var_36 = 2
    var_37 = 0
    var_38 = [var_36, var_37]
    var_39 = True
    var_40 = module_0.get_in(var_38, var_26, no_default=var_39)
    var_41 = {var_37: var_39}
    var_42 = {var_38: var_21}
    var_43 = [var_41, var_42]
    var_44 = {var_36: var_43}
    var_45 = [var_36, var_13, var_37]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 1
    var_47 = [var_36, var_39, var_38]
    var_48 = module_0.get_in(var_47, var_44)
    assert var_48 == 2
    var_49 = [var_36, var_21, var_9]
    var_50 = module_0.get_in(var_49, var_44)
    assert var_50 is None
    var_51 = 'All tests passed!'
    var_52 = print(var_51)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 2
    var_16 = 3
    var_17 = [var_3, var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = [var_18, var_19, var_20]
    var_22 = [var_17, var_21]
    var_23 = [var_13, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3, var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 6
    var_27 = [var_15, var_13]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = {var_1: var_3}
    var_30 = {var_2: var_15}
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_13, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 1
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 2
    var_37 = [var_0, var_15, var_9]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = {var_1: var_3}
    var_40 = {var_0: var_39}
    var_41 = 'a'
    var_42 = 'c'
    var_43 = [var_41, var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_40, no_default=var_44)
    var_46 = {var_42: var_44}
    var_47 = {var_41: var_46}
    var_48 = [var_41, var_43]
    var_49 = module_0.get_in(var_48, var_47, var_13)
    assert var_49 == 0
    var_50 = {var_41: var_44}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = {var_42: var_44}
    var_54 = {var_41: var_53}
    var_55 = [var_41, var_43]
    var_56 = None
    var_57 = module_0.get_in(var_55, var_54, var_56)
    assert var_57 is None
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = []
    var_14 = module_0.get_in(var_13, var_6)
    var_15 = 'd'
    var_16 = [var_0, var_1, var_15]
    var_17 = 2
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 2
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 'z'
    var_22 = [var_19, var_20, var_21]
    var_23 = 3
    var_24 = module_0.get_in(var_22, var_6, var_23)
    assert var_24 == 3
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'd'
    var_28 = [var_25, var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = [var_28, var_17, var_23]
    var_32 = 4
    var_33 = 5
    var_34 = 6
    var_35 = [var_32, var_33, var_34]
    var_36 = [var_31, var_35]
    var_37 = 0
    var_38 = [var_37, var_28]
    var_39 = module_0.get_in(var_38, var_36)
    assert var_39 == 2
    var_40 = [var_28, var_17]
    var_41 = module_0.get_in(var_40, var_36)
    assert var_41 == 6
    var_42 = {var_26: var_28}
    var_43 = {var_27: var_17}
    var_44 = [var_42, var_43]
    var_45 = {var_25: var_44}
    var_46 = [var_25, var_37, var_26]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 1
    var_48 = [var_25, var_28, var_27]
    var_49 = module_0.get_in(var_48, var_45)
    assert var_49 == 2
    var_50 = 'All tests passed!'
    var_51 = print(var_50)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = {var_1: var_3}
    var_16 = 2
    var_17 = {var_2: var_16}
    var_18 = [var_15, var_17]
    var_19 = {var_0: var_18}
    var_20 = [var_0, var_13, var_1]
    var_21 = module_0.get_in(var_20, var_19)
    assert var_21 == 1
    var_22 = [var_0, var_3, var_2]
    var_23 = module_0.get_in(var_22, var_19)
    assert var_23 == 2
    var_24 = [var_0, var_16]
    var_25 = module_0.get_in(var_24, var_19)
    assert var_25 is None
    var_26 = {var_1: var_3}
    var_27 = {var_0: var_26}
    var_28 = 'a'
    var_29 = 'c'
    var_30 = [var_28, var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_27, no_default=var_31)
    var_33 = {var_30: var_31}
    var_34 = {var_29: var_33}
    var_35 = [var_34]
    var_36 = {var_28: var_35}
    var_37 = [var_28, var_13, var_29, var_30]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 1
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = 10
    var_34 = [var_26, var_29, var_33]
    var_35 = module_0.get_in(var_34, var_14)
    assert var_35 is None
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = []
    var_34 = module_0.get_in(var_33, var_14)
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'd'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = []
    var_59 = {var_52: var_55}
    var_60 = module_0.get_in(var_58, var_59)
    var_61 = []
    var_62 = [var_55, var_21, var_23]
    var_63 = module_0.get_in(var_61, var_62)
    var_64 = 'All tests passed!'
    var_65 = print(var_64)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = []
    var_34 = module_0.get_in(var_33, var_14)
    var_35 = 'name'
    var_36 = 'first'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_14, no_default=var_38)
    var_40 = 'All tests passed!'
    var_41 = print(var_40)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = 10
    var_34 = [var_26, var_29, var_33]
    var_35 = module_0.get_in(var_34, var_14)
    assert var_35 is None
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_36, var_37]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 6
    var_51 = [var_37, var_15]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 7
    var_53 = [var_15, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_15, var_38]
    var_56 = -1
    var_57 = module_0.get_in(var_55, var_48, var_56)
    assert var_57 == -1
    var_58 = 3
    var_59 = 0
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_48, no_default=var_61)
    var_63 = 'a'
    var_64 = 'd'
    var_65 = 'b'
    var_66 = {var_65: var_36}
    var_67 = 'c'
    var_68 = {var_67: var_37}
    var_69 = [var_66, var_68]
    var_70 = 'e'
    var_71 = [var_38, var_40, var_41]
    var_72 = {var_70: var_71}
    var_73 = {var_63: var_69, var_64: var_72}
    var_74 = [var_63, var_15, var_65]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 == 1
    var_76 = [var_63, var_36, var_67]
    var_77 = module_0.get_in(var_76, var_73)
    assert var_77 == 2
    var_78 = [var_64, var_70, var_36]
    var_79 = module_0.get_in(var_78, var_73)
    assert var_79 == 4
    var_80 = [var_64, var_70, var_38]
    var_81 = module_0.get_in(var_80, var_73)
    assert var_81 is None
    var_82 = [var_64, var_70, var_38]
    var_83 = -1
    var_84 = module_0.get_in(var_82, var_73, var_83)
    assert var_84 == -1
    var_85 = 'd'
    var_86 = 'f'
    var_87 = [var_85, var_86]
    var_88 = True
    var_89 = module_0.get_in(var_87, var_73, no_default=var_88)
    var_90 = 'All tests passed!'
    var_91 = print(var_90)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'd'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = []
    var_14 = module_0.get_in(var_13, var_6)
    var_15 = 'd'
    var_16 = [var_0, var_1, var_15]
    var_17 = module_0.get_in(var_16, var_6)
    assert var_17 is None
    var_18 = [var_0, var_1, var_15]
    var_19 = 0
    var_20 = module_0.get_in(var_18, var_6, var_19)
    assert var_20 == 0
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'd'
    var_24 = [var_21, var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_6, no_default=var_25)
    var_27 = 2
    var_28 = [var_24, var_27]
    var_29 = 3
    var_30 = 4
    var_31 = [var_29, var_30]
    var_32 = [var_28, var_31]
    var_33 = [var_19, var_24]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_24, var_19]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 3
    var_37 = [var_19]
    var_38 = module_0.get_in(var_37, var_32)
    var_39 = []
    var_40 = module_0.get_in(var_39, var_32)
    var_41 = [var_19, var_27]
    var_42 = module_0.get_in(var_41, var_32)
    assert var_42 is None
    var_43 = [var_19, var_27]
    var_44 = module_0.get_in(var_43, var_32, var_19)
    assert var_44 == 0
    var_45 = 0
    var_46 = 2
    var_47 = [var_45, var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_32, no_default=var_48)
    var_50 = {var_46: var_48}
    var_51 = {var_47: var_27}
    var_52 = [var_50, var_51]
    var_53 = {var_45: var_52}
    var_54 = [var_45, var_19, var_46]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 == 1
    var_56 = [var_45, var_48, var_47]
    var_57 = module_0.get_in(var_56, var_53)
    assert var_57 == 2
    var_58 = [var_45, var_19]
    var_59 = module_0.get_in(var_58, var_53)
    var_60 = [var_45]
    var_61 = module_0.get_in(var_60, var_53)
    var_62 = []
    var_63 = module_0.get_in(var_62, var_53)
    var_64 = [var_45, var_27]
    var_65 = module_0.get_in(var_64, var_53)
    assert var_65 is None
    var_66 = [var_45, var_27]
    var_67 = module_0.get_in(var_66, var_53, var_19)
    assert var_67 == 0
    var_68 = 'a'
    var_69 = 2
    var_70 = [var_68, var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_53, no_default=var_71)
    var_73 = {var_68: var_71}
    var_74 = [var_69]
    var_75 = module_0.get_in(var_74, var_73, var_27)
    assert var_75 == 2
    var_76 = [var_69]
    var_77 = None
    var_78 = module_0.get_in(var_76, var_73, var_77)
    assert var_78 is None
    var_79 = [var_69]
    var_80 = []
    var_81 = module_0.get_in(var_79, var_73, var_80)
    var_82 = {var_68: var_71}
    var_83 = 'b'
    var_84 = [var_83]
    var_85 = True
    var_86 = module_0.get_in(var_84, var_82, no_default=var_85)
    var_87 = {var_83: var_86}
    var_88 = []
    var_89 = module_0.get_in(var_88, var_87)
    var_90 = []
    var_91 = module_0.get_in(var_90, var_87, var_27)
    var_92 = []
    var_93 = True
    var_94 = module_0.get_in(var_92, var_87, no_default=var_93)
    var_95 = 'All tests passed!'
    var_96 = print(var_95)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 'd'
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_6)
    assert var_11 is None
    var_12 = [var_0, var_1, var_9]
    var_13 = 0
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 0
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = [var_18, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_13, var_18]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_18, var_18]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 4
    var_31 = [var_21, var_13]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21, var_13]
    var_34 = module_0.get_in(var_33, var_26, var_13)
    assert var_34 == 0
    var_35 = 2
    var_36 = 0
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_26, no_default=var_38)
    var_40 = {var_36: var_38}
    var_41 = {var_37: var_21}
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_42}
    var_44 = [var_35, var_13, var_36]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 1
    var_46 = [var_35, var_38, var_37]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_35, var_21, var_9]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'd'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0
    var_31 = 'y'
    var_32 = [var_31]
    var_33 = {}
    var_34 = True
    var_35 = module_0.get_in(var_32, var_33, no_default=var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = 4
    var_41 = 5
    var_42 = 6
    var_43 = [var_40, var_41, var_42]
    var_44 = 7
    var_45 = 8
    var_46 = 9
    var_47 = [var_44, var_45, var_46]
    var_48 = [var_39, var_43, var_47]
    var_49 = [var_36, var_37]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 6
    var_51 = [var_37, var_15]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 7
    var_53 = [var_15, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_38, var_15]
    var_56 = module_0.get_in(var_55, var_48)
    assert var_56 is None
    var_57 = (var_36, var_37, var_38)
    var_58 = (var_40, var_41, var_42)
    var_59 = (var_44, var_45, var_46)
    var_60 = (var_57, var_58, var_59)
    var_61 = [var_36, var_37]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 == 6
    var_63 = [var_37, var_15]
    var_64 = module_0.get_in(var_63, var_60)
    assert var_64 == 7
    var_65 = [var_15, var_38]
    var_66 = module_0.get_in(var_65, var_60)
    assert var_66 is None
    var_67 = [var_38, var_15]
    var_68 = module_0.get_in(var_67, var_60)
    assert var_68 is None
    var_69 = 'a'
    var_70 = 'c'
    var_71 = 'b'
    var_72 = {var_71: var_38}
    var_73 = [var_36, var_37, var_72]
    var_74 = [var_42, var_44]
    var_75 = (var_40, var_41, var_74)
    var_76 = {var_69: var_73, var_70: var_75}
    var_77 = [var_69, var_37, var_71]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 == 3
    var_79 = [var_70, var_37, var_36]
    var_80 = module_0.get_in(var_79, var_76)
    assert var_80 == 7
    var_81 = [var_69, var_38]
    var_82 = module_0.get_in(var_81, var_76)
    assert var_82 is None
    var_83 = [var_70, var_38]
    var_84 = module_0.get_in(var_83, var_76)
    assert var_84 is None
    var_85 = 'All tests passed!'
    var_86 = print(var_85)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = [var_1, var_20]
    var_24 = module_0.get_in(var_23, var_14, var_15)
    assert var_24 == 0
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = 1
    var_31 = [var_26, var_5, var_30]
    var_32 = module_0.get_in(var_31, var_14)
    var_33 = 10
    var_34 = [var_26, var_29, var_33]
    var_35 = module_0.get_in(var_34, var_14)
    assert var_35 is None
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



