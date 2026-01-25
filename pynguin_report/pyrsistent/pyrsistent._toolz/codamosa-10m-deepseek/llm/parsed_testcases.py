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
    var_44 = [var_15, var_31]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 2
    var_46 = [var_32, var_32]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 9
    var_48 = [var_31, var_33]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = 'a'
    var_51 = 'd'
    var_52 = 'b'
    var_53 = {var_52: var_31}
    var_54 = 'c'
    var_55 = {var_54: var_32}
    var_56 = [var_53, var_55]
    var_57 = 'e'
    var_58 = [var_33, var_35, var_36]
    var_59 = {var_57: var_58}
    var_60 = {var_50: var_56, var_51: var_59}
    var_61 = [var_50, var_15, var_52]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 == 1
    var_63 = [var_51, var_57, var_31]
    var_64 = module_0.get_in(var_63, var_60)
    assert var_64 == 4
    var_65 = [var_50, var_31, var_51]
    var_66 = module_0.get_in(var_65, var_60)
    assert var_66 is None
    var_67 = 'x'
    var_68 = 'y'
    var_69 = [var_67, var_68]
    var_70 = {}
    var_71 = 'default'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'default'
    var_73 = [var_67, var_68]
    var_74 = {}
    var_75 = True
    var_76 = module_0.get_in(var_73, var_74, no_default=var_75)
    var_77 = 'All test cases passed!'
    var_78 = print(var_77)



# Parsed testcases at query #2
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
    var_49 = [var_15, var_36]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 2
    var_51 = [var_37, var_37]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 9
    var_53 = [var_36, var_38]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 is None
    var_55 = [var_36, var_38]
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
    var_71 = [var_59, var_65, var_37]
    var_72 = module_0.get_in(var_71, var_68)
    assert var_72 == 5
    var_73 = [var_58, var_36, var_59]
    var_74 = module_0.get_in(var_73, var_68)
    assert var_74 is None
    var_75 = 'All tests passed!'
    var_76 = print(var_75)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

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
    var_36 = module_0.get_in(var_35, var_28, var_13)
    assert var_36 == 0
    var_37 = 2
    var_38 = 0
    var_39 = [var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_28, no_default=var_40)
    var_42 = (var_40, var_21, var_22)
    var_43 = (var_24, var_25, var_26)
    var_44 = (var_42, var_43)
    var_45 = [var_13, var_40]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 2
    var_47 = [var_40, var_21]
    var_48 = module_0.get_in(var_47, var_44)
    assert var_48 == 6
    var_49 = [var_21, var_13]
    var_50 = module_0.get_in(var_49, var_44)
    assert var_50 is None
    var_51 = [var_21, var_13]
    var_52 = module_0.get_in(var_51, var_44, var_13)
    assert var_52 == 0
    var_53 = 2
    var_54 = 0
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_44, no_default=var_56)
    var_58 = {var_54: var_56}
    var_59 = {var_55: var_21}
    var_60 = [var_58, var_59]
    var_61 = {var_53: var_60}
    var_62 = [var_53, var_13, var_54]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 == 1
    var_64 = [var_53, var_56, var_55]
    var_65 = module_0.get_in(var_64, var_61)
    assert var_65 == 2
    var_66 = [var_53, var_21, var_9]
    var_67 = module_0.get_in(var_66, var_61)
    assert var_67 is None
    var_68 = [var_53, var_21, var_9]
    var_69 = module_0.get_in(var_68, var_61, var_13)
    assert var_69 == 0
    var_70 = 'a'
    var_71 = 2
    var_72 = 'd'
    var_73 = [var_70, var_71, var_72]
    var_74 = True
    var_75 = module_0.get_in(var_73, var_61, no_default=var_74)
    var_76 = 'All tests passed!'
    var_77 = print(var_76)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

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
    var_62 = module_0.get_in(var_61, var_6)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_28)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_46)
    var_67 = 'All tests passed!'
    var_68 = print(var_67)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

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
    var_36 = module_0.get_in(var_35, var_28, var_13)
    assert var_36 == 0
    var_37 = 2
    var_38 = 0
    var_39 = [var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_28, no_default=var_40)
    var_42 = {var_38: var_40}
    var_43 = {var_39: var_21}
    var_44 = [var_42, var_43]
    var_45 = {var_37: var_44}
    var_46 = [var_37, var_13, var_38]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 1
    var_48 = [var_37, var_40, var_39]
    var_49 = module_0.get_in(var_48, var_45)
    assert var_49 == 2
    var_50 = [var_37, var_21, var_9]
    var_51 = module_0.get_in(var_50, var_45)
    assert var_51 is None
    var_52 = [var_37, var_21, var_9]
    var_53 = module_0.get_in(var_52, var_45, var_13)
    assert var_53 == 0
    var_54 = 'a'
    var_55 = 2
    var_56 = 'd'
    var_57 = [var_54, var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_45, no_default=var_58)
    var_60 = 'All tests passed!'
    var_61 = print(var_60)



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

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
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #10
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
    var_50 = 'a'
    var_51 = 'c'
    var_52 = 'b'
    var_53 = {var_52: var_33}
    var_54 = [var_31, var_32, var_53]
    var_55 = 'd'
    var_56 = [var_35, var_36]
    var_57 = {var_55: var_56}
    var_58 = {var_50: var_54, var_51: var_57}
    var_59 = [var_50, var_32, var_52]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 == 3
    var_61 = [var_51, var_55, var_31]
    var_62 = module_0.get_in(var_61, var_58)
    assert var_62 == 5
    var_63 = [var_50, var_33]
    var_64 = module_0.get_in(var_63, var_58)
    assert var_64 is None
    var_65 = 'x'
    var_66 = 'y'
    var_67 = [var_65, var_66]
    var_68 = {}
    var_69 = 'default'
    var_70 = module_0.get_in(var_67, var_68, var_69)
    assert var_70 == 'default'
    var_71 = [var_65, var_66]
    var_72 = {}
    var_73 = None
    var_74 = module_0.get_in(var_71, var_72, var_73)
    assert var_74 is None
    var_75 = 'y'
    var_76 = [var_75]
    var_77 = {}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)
    var_80 = {var_51: var_31}
    var_81 = {var_52: var_80}
    var_82 = {var_50: var_81}
    var_83 = [var_50, var_52, var_51]
    var_84 = module_0.get_in(var_83, var_82, var_15)
    assert var_84 == 1
    var_85 = [var_50, var_52, var_55]
    var_86 = module_0.get_in(var_85, var_82, var_15)
    assert var_86 == 0
    var_87 = 'All tests passed!'
    var_88 = print(var_87)



# Parsed testcases at query #11
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
    var_81 = []
    var_82 = [var_36, var_37, var_38]
    var_83 = module_0.get_in(var_81, var_82)
    var_84 = 'x'
    var_85 = [var_84]
    var_86 = {}
    var_87 = 'Default'
    var_88 = module_0.get_in(var_85, var_86, var_87)
    assert var_88 == 'Default'
    var_89 = 'x'
    var_90 = [var_89]
    var_91 = {}
    var_92 = True
    var_93 = module_0.get_in(var_90, var_91, no_default=var_92)
    var_94 = 'All tests passed!'
    var_95 = print(var_94)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

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
    var_58 = 'All tests passed.'
    var_59 = print(var_58)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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
    var_48 = [var_35, var_21, var_37]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_37]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'c'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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



# Parsed testcases at query #16
#--------------------------


import pyrsistent._toolz as module_0

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
    var_62 = module_0.get_in(var_61, var_6)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_28)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_46)
    var_67 = 'x'
    var_68 = 'y'
    var_69 = 'z'
    var_70 = [var_67, var_68, var_69]
    var_71 = {}
    var_72 = 'not found'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'not found'
    var_74 = [var_13, var_58, var_21]
    var_75 = []
    var_76 = module_0.get_in(var_74, var_75, var_72)
    assert var_76 == 'not found'
    var_77 = 'All tests passed!'
    var_78 = print(var_77)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._toolz as module_0

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
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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
    var_48 = [var_35, var_21, var_37]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_37]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'c'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #19
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
    var_76 = [var_64, var_70, var_36]
    var_77 = module_0.get_in(var_76, var_73)
    assert var_77 == 4
    var_78 = [var_63, var_36, var_64]
    var_79 = module_0.get_in(var_78, var_73)
    assert var_79 is None
    var_80 = 'f'
    var_81 = [var_64, var_80]
    var_82 = 'No key'
    var_83 = module_0.get_in(var_81, var_73, var_82)
    assert var_83 == 'No key'
    var_84 = 'a'
    var_85 = 2
    var_86 = 'b'
    var_87 = [var_84, var_85, var_86]
    var_88 = True
    var_89 = module_0.get_in(var_87, var_73, no_default=var_88)
    var_90 = 'All tests passed!'
    var_91 = print(var_90)



# Parsed testcases at query #20
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
    var_50 = [var_15, var_33]
    var_51 = 'Not Found'
    var_52 = module_0.get_in(var_50, var_43, var_51)
    assert var_52 == 'Not Found'
    var_53 = 'a'
    var_54 = 'c'
    var_55 = 'b'
    var_56 = {var_55: var_33}
    var_57 = [var_31, var_32, var_56]
    var_58 = 'd'
    var_59 = [var_35, var_36]
    var_60 = {var_58: var_59}
    var_61 = {var_53: var_57, var_54: var_60}
    var_62 = [var_53, var_32, var_55]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 == 3
    var_64 = [var_54, var_58, var_31]
    var_65 = module_0.get_in(var_64, var_61)
    assert var_65 == 5
    var_66 = [var_53, var_33]
    var_67 = module_0.get_in(var_66, var_61)
    assert var_67 is None
    var_68 = 'e'
    var_69 = [var_54, var_68]
    var_70 = 'Default'
    var_71 = module_0.get_in(var_69, var_61, var_70)
    assert var_71 == 'Default'
    var_72 = 'y'
    var_73 = [var_72]
    var_74 = {}
    var_75 = True
    var_76 = module_0.get_in(var_73, var_74, no_default=var_75)
    var_77 = 'x'
    var_78 = 'y'
    var_79 = [var_77, var_78]
    var_80 = {}
    var_81 = module_0.get_in(var_79, var_80, var_70)
    assert var_81 == 'Default'
    var_82 = 'All tests passed!'
    var_83 = print(var_82)



# Parsed testcases at query #21
#--------------------------


import pyrsistent._toolz as module_0

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
    var_19 = [var_0, var_15]
    var_20 = module_0.get_in(var_19, var_6, var_17)
    assert var_20 == 2
    var_21 = [var_15]
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 2
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 'd'
    var_26 = [var_23, var_24, var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_6, no_default=var_27)
    var_29 = 3
    var_30 = 4
    var_31 = [var_29, var_30]
    var_32 = [var_26, var_17, var_31]
    var_33 = 0
    var_34 = [var_17, var_33]
    var_35 = module_0.get_in(var_34, var_32)
    assert var_35 == 3
    var_36 = [var_17, var_26]
    var_37 = module_0.get_in(var_36, var_32)
    assert var_37 == 4
    var_38 = [var_33]
    var_39 = module_0.get_in(var_38, var_32)
    assert var_39 == 1
    var_40 = {var_24: var_29}
    var_41 = [var_26, var_17, var_40]
    var_42 = {var_23: var_41}
    var_43 = [var_23, var_17, var_24]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 3
    var_45 = [var_23, var_26]
    var_46 = module_0.get_in(var_45, var_42)
    assert var_46 == 2
    var_47 = []
    var_48 = module_0.get_in(var_47, var_42)
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._toolz as module_0

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



# Parsed testcases at query #23
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
    var_78 = [var_64, var_70, var_37]
    var_79 = module_0.get_in(var_78, var_73)
    assert var_79 == 5
    var_80 = [var_64, var_70, var_38]
    var_81 = module_0.get_in(var_80, var_73)
    assert var_81 is None
    var_82 = [var_63, var_37, var_65]
    var_83 = module_0.get_in(var_82, var_73, var_56)
    assert var_83 == 'Not Found'
    var_84 = 'a'
    var_85 = 2
    var_86 = 'b'
    var_87 = [var_84, var_85, var_86]
    var_88 = True
    var_89 = module_0.get_in(var_87, var_73, no_default=var_88)
    var_90 = 'All tests passed!'
    var_91 = print(var_90)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

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
    var_53 = [var_41]
    var_54 = None
    var_55 = module_0.get_in(var_53, var_54)
    assert var_55 is None
    var_56 = [var_41]
    var_57 = module_0.get_in(var_56, var_54, var_13)
    assert var_57 == 0
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

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



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

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
    var_67 = 'All tests passed!'
    var_68 = print(var_67)



# Parsed testcases at query #5
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
    var_58 = 'Not Found'
    var_59 = module_0.get_in(var_57, var_48, var_58)
    assert var_59 == 'Not Found'
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
    var_75 = [var_60, var_36, var_61]
    var_76 = module_0.get_in(var_75, var_70)
    assert var_76 is None
    var_77 = 'f'
    var_78 = [var_61, var_77]
    var_79 = 'Missing'
    var_80 = module_0.get_in(var_78, var_70, var_79)
    assert var_80 == 'Missing'
    var_81 = []
    var_82 = module_0.get_in(var_81, var_14)
    var_83 = []
    var_84 = module_0.get_in(var_83, var_48)
    var_85 = 'All tests passed!'
    var_86 = print(var_85)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

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



# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

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
    var_13 = 'd'
    var_14 = [var_0, var_1, var_2, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_1, var_2, var_13]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = 'd'
    var_23 = [var_19, var_20, var_21, var_22]
    var_24 = True
    var_25 = module_0.get_in(var_23, var_6, no_default=var_24)
    var_26 = 2
    var_27 = 3
    var_28 = [var_22, var_26, var_27]
    var_29 = 4
    var_30 = 5
    var_31 = 6
    var_32 = [var_29, var_30, var_31]
    var_33 = [var_28, var_32]
    var_34 = [var_17, var_22]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_22, var_26]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 == 6
    var_38 = [var_17, var_27]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_17, var_27]
    var_41 = -1
    var_42 = module_0.get_in(var_40, var_33, var_41)
    assert var_42 == -1
    var_43 = 0
    var_44 = 3
    var_45 = [var_43, var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_33, no_default=var_46)
    var_48 = {var_44: var_46}
    var_49 = {var_45: var_26}
    var_50 = [var_48, var_49]
    var_51 = {var_43: var_50}
    var_52 = [var_43, var_17, var_44]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 == 1
    var_54 = [var_43, var_46, var_45]
    var_55 = module_0.get_in(var_54, var_51)
    assert var_55 == 2
    var_56 = [var_43, var_26, var_13]
    var_57 = module_0.get_in(var_56, var_51)
    assert var_57 is None
    var_58 = [var_43, var_26, var_13]
    var_59 = module_0.get_in(var_58, var_51, var_17)
    assert var_59 == 0
    var_60 = 'a'
    var_61 = 2
    var_62 = 'd'
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_51, no_default=var_64)
    var_66 = {var_60: var_63}
    var_67 = []
    var_68 = module_0.get_in(var_67, var_66)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_66, var_17)
    var_71 = []
    var_72 = True
    var_73 = module_0.get_in(var_71, var_66, no_default=var_72)
    var_74 = {var_71: var_63}
    var_75 = [var_72]
    var_76 = module_0.get_in(var_75, var_74)
    assert var_76 is None
    var_77 = [var_72]
    var_78 = module_0.get_in(var_77, var_74, var_17)
    assert var_78 == 0
    var_79 = [var_72]
    var_80 = 'missing'
    var_81 = module_0.get_in(var_79, var_74, var_80)
    assert var_81 == 'missing'
    var_82 = {var_71: var_63}
    var_83 = 'b'
    var_84 = [var_83]
    var_85 = True
    var_86 = module_0.get_in(var_84, var_82, no_default=var_85)
    var_87 = {var_84: var_86}
    var_88 = {var_83: var_87}
    var_89 = [var_83, var_85]
    var_90 = module_0.get_in(var_89, var_88)
    assert var_90 is None
    var_91 = [var_83, var_85]
    var_92 = module_0.get_in(var_91, var_88, var_17)
    assert var_92 == 0
    var_93 = [var_83, var_85, var_13]
    var_94 = module_0.get_in(var_93, var_88, var_80)
    assert var_94 == 'missing'
    var_95 = 'All tests passed!'
    var_96 = print(var_95)



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

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
    var_13 = 'not found'
    var_14 = module_0.get_in(var_12, var_6, var_13)
    assert var_14 == 'not found'
    var_15 = 1
    var_16 = {var_1: var_15}
    var_17 = 2
    var_18 = {var_2: var_17}
    var_19 = [var_16, var_18]
    var_20 = {var_0: var_19}
    var_21 = 0
    var_22 = [var_0, var_21, var_1]
    var_23 = module_0.get_in(var_22, var_20)
    assert var_23 == 1
    var_24 = [var_0, var_15, var_2]
    var_25 = module_0.get_in(var_24, var_20)
    assert var_25 == 2
    var_26 = [var_0, var_17]
    var_27 = module_0.get_in(var_26, var_20)
    assert var_27 is None
    var_28 = 3
    var_29 = [var_15, var_17, var_28]
    var_30 = {var_1: var_29}
    var_31 = {var_0: var_30}
    var_32 = [var_0, var_1, var_15]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 2
    var_34 = 'x'
    var_35 = [var_34]
    var_36 = {}
    var_37 = True
    var_38 = module_0.get_in(var_35, var_36, no_default=var_37)
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

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
    var_58 = 'All tests passed.'
    var_59 = print(var_58)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0

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
    var_27 = [var_16, var_11]
    var_28 = 5
    var_29 = module_0.get_in(var_27, var_23, var_28)
    assert var_29 == 5
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_23, no_default=var_33)
    var_35 = {var_31: var_33}
    var_36 = {var_32: var_11}
    var_37 = [var_35, var_36]
    var_38 = {var_30: var_37}
    var_39 = [var_30, var_24, var_31]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 1
    var_41 = [var_30, var_33, var_32]
    var_42 = module_0.get_in(var_41, var_38)
    assert var_42 == 2
    var_43 = [var_30, var_11, var_9]
    var_44 = module_0.get_in(var_43, var_38, var_20)
    assert var_44 == 3
    var_45 = 'All tests passed'
    var_46 = print(var_45)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

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



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

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
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'd'
    var_18 = [var_15, var_16, var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = 4
    var_26 = 5
    var_27 = 6
    var_28 = [var_25, var_26, var_27]
    var_29 = [var_24, var_28]
    var_30 = [var_13, var_21]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 2
    var_32 = [var_21, var_22]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 6
    var_34 = [var_22, var_13]
    var_35 = module_0.get_in(var_34, var_29)
    assert var_35 is None
    var_36 = [var_22, var_13]
    var_37 = -1
    var_38 = module_0.get_in(var_36, var_29, var_37)
    assert var_38 == -1
    var_39 = 2
    var_40 = 0
    var_41 = [var_39, var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_29, no_default=var_42)
    var_44 = {var_40: var_21}
    var_45 = {var_41: var_22}
    var_46 = [var_44, var_45]
    var_47 = {var_39: var_46}
    var_48 = [var_39, var_13, var_40]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 1
    var_50 = [var_39, var_21, var_41]
    var_51 = module_0.get_in(var_50, var_47)
    assert var_51 == 2
    var_52 = [var_39, var_22, var_9]
    var_53 = module_0.get_in(var_52, var_47)
    assert var_53 is None
    var_54 = 'All tests passed!'
    var_55 = print(var_54)



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import pyrsistent._toolz as module_0

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
    var_29 = [var_18, var_13]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
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
    var_48 = [var_35, var_21, var_37]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_35, var_21, var_37]
    var_51 = module_0.get_in(var_50, var_43, var_13)
    assert var_51 == 0
    var_52 = 'a'
    var_53 = 2
    var_54 = 'c'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_43, no_default=var_56)
    var_58 = 'All tests passed'
    var_59 = print(var_58)



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


import pyrsistent._toolz as module_0

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
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'd'
    var_27 = [var_24, var_25, var_26]
    var_28 = True
    var_29 = module_0.get_in(var_27, var_19, no_default=var_28)
    var_30 = 'All tests passed'
    var_31 = print(var_30)



# Parsed testcases at query #24
#--------------------------


import pyrsistent._toolz as module_0

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



# Parsed testcases at query #25
#--------------------------


import pyrsistent._toolz as module_0

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
    var_27 = [var_16, var_11]
    var_28 = 5
    var_29 = module_0.get_in(var_27, var_23, var_28)
    assert var_29 == 5
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_23, no_default=var_33)
    var_35 = {var_31: var_33}
    var_36 = {var_32: var_11}
    var_37 = [var_35, var_36]
    var_38 = {var_30: var_37}
    var_39 = [var_30, var_24, var_31]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 1
    var_41 = [var_30, var_33, var_32]
    var_42 = module_0.get_in(var_41, var_38)
    assert var_42 == 2
    var_43 = [var_30, var_11, var_9]
    var_44 = module_0.get_in(var_43, var_38, var_20)
    assert var_44 == 3
    var_45 = 'All tests passed.'
    var_46 = print(var_45)



